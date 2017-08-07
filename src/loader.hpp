/*
 Copyright 2017 Nervana Systems Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#pragma once

#include <vector>
#include <cstdio>
#include <iostream>
#include <chrono>
#include <utility>
#include <algorithm>
#include <map>
#include <future>

#include "manifest.hpp"
#include "provider_factory.hpp"

#include "async_manager.hpp"
#include "buffer_batch.hpp"
#include "batch_iterator.hpp"
#include "batch_decoder.hpp"
#include "block_loader_file.hpp"
#include "block_manager.hpp"
#include "log.hpp"
#include "util.hpp"
#include "web_app.hpp"

namespace nervana
{
    class loader_config;
    class loader;
    class loader_factory;
    class loader_local;
    class dataset_builder;
    class batch_decoder;
}

class nervana::loader_config : public nervana::interface::config
{
public:
    std::string manifest_filename;
    std::string manifest_root;
    int         batch_size;

    std::string                 cache_directory      = "";
    int                         block_size           = 0;
    float                       subset_fraction      = 1.0;
    bool                        shuffle_enable       = false;
    bool                        shuffle_manifest     = false;
    bool                        pinned               = false;
    int                         random_seed          = 0;
    uint32_t                    decode_thread_count  = 0;
    std::string                 iteration_mode       = "ONCE";
    int                         iteration_mode_count = 0;
    uint16_t                    web_server_port      = 0;
    std::vector<nlohmann::json> etl;
    std::vector<nlohmann::json> augmentation;

    loader_config(nlohmann::json js);

private:
    loader_config() {}
    std::vector<std::shared_ptr<nervana::interface::config_info_interface>> config_list = {
        ADD_SCALAR(manifest_filename, mode::REQUIRED),
        ADD_SCALAR(manifest_root, mode::OPTIONAL),
        ADD_SCALAR(batch_size, mode::REQUIRED),
        ADD_SCALAR(cache_directory, mode::OPTIONAL),
        ADD_SCALAR(block_size, mode::OPTIONAL),
        ADD_SCALAR(subset_fraction,
                   mode::OPTIONAL,
                   [](decltype(subset_fraction) v) { return v <= 1.0 && v >= 0.0; }),
        ADD_SCALAR(shuffle_enable, mode::OPTIONAL),
        ADD_SCALAR(shuffle_manifest, mode::OPTIONAL),
        ADD_SCALAR(decode_thread_count, mode::OPTIONAL),
        ADD_SCALAR(pinned, mode::OPTIONAL),
        ADD_SCALAR(random_seed, mode::OPTIONAL),
        ADD_SCALAR(iteration_mode, mode::OPTIONAL),
        ADD_SCALAR(iteration_mode_count, mode::OPTIONAL),
        ADD_SCALAR(web_server_port, mode::OPTIONAL),
        ADD_OBJECT(etl, mode::REQUIRED),
        ADD_OBJECT(augmentation, mode::OPTIONAL)};

    void validate();
};

// this is interface for loaders
class nervana::loader
{
public:
    enum class BatchMode
    {
        INFINITE,
        ONCE,
        COUNT
    };

    // member typedefs provided through inheriting from std::iterator
    class iterator : public std::iterator<std::input_iterator_tag, // iterator_category
                                          fixed_buffer_map         // value_type
                                          >
    {
        friend class loader_local;
        friend class loader_remote;

    public:
        explicit iterator(loader& ld, bool is_end);
        iterator(const iterator&);
        ~iterator() {}
        iterator& operator++();
        iterator& operator++(int);
        bool operator==(const iterator& other) const;
        bool operator!=(const iterator& other) const;
        const fixed_buffer_map& operator*() const;
        const size_t& position() const { return m_current_loader.position(); }
        bool          positional_end() const;

    private:
        iterator() = delete;

        loader&          m_current_loader;
        const bool       m_is_end;
    };

    virtual ~loader() {}
    virtual std::vector<std::string> get_buffer_names() const = 0;
    virtual std::map<std::string, shape_type> get_names_and_shapes() const = 0;
    virtual shape_t get_shape(const std::string& name) const = 0;

    virtual int record_count() const = 0;
    virtual int batch_size() const   = 0;
    virtual int batch_count() const  = 0;

    virtual iterator  begin()            = 0;
    virtual iterator  end()              = 0;
    virtual iterator& get_current_iter() = 0;
    virtual iterator& get_end_iter()     = 0;

    virtual const fixed_buffer_map* get_output_buffer() const  = 0;
    virtual const size_t&           position()                 = 0;
    virtual void                    reset()                    = 0;
    virtual nlohmann::json          get_current_config() const = 0;

protected:
    virtual void increment_position() = 0;
};

class nervana::loader_factory
{
public:
    std::unique_ptr<loader> get_loader(const std::string& config);
    std::unique_ptr<loader> get_loader(const nlohmann::json& config);

private:
    bool remote_version(const nlohmann::json& config);
};

class nervana::loader_local final : public nervana::loader
{
public:
    loader_local(const std::string&);
    loader_local(const nlohmann::json&);

    ~loader_local() override;

    std::vector<std::string> get_buffer_names() const override;
    std::map<std::string, shape_type> get_names_and_shapes() const override;
    shape_t get_shape(const std::string& name) const override;

    int      record_count() const override { return m_manifest->record_count(); }
    int      batch_size() const override { return m_batch_size; }
    int      batch_count() const override { return m_batch_count_value; }
    iterator begin() override
    {
        reset();
        return m_current_iter;
    }

    iterator end() override { return m_end_iter; }
    // These are returning references
    iterator&               get_current_iter() override { return m_current_iter; }
    iterator&               get_end_iter() override { return m_end_iter; }
    const fixed_buffer_map* get_output_buffer() const override { return m_output_buffer_ptr; }
    const size_t&           position() override { return m_position; }
    void                    reset() override
    {
        m_final_stage->reset();
        m_output_buffer_ptr = m_final_stage->next();
        m_position          = 0;
    }

    nlohmann::json get_current_config() const override { return m_current_config; }
private:
    friend class nervana::loader::iterator;

    loader_local() = delete;
    void initialize(const nlohmann::json& config_json);
    void increment_position() override;

    iterator                                                m_current_iter;
    iterator                                                m_end_iter;
    std::shared_ptr<manifest_file>                          m_manifest;
    std::shared_ptr<block_loader_file>                      m_block_loader;
    std::shared_ptr<block_manager>                          m_block_manager;
    std::shared_ptr<batch_iterator>                         m_batch_iterator;
    std::shared_ptr<provider_interface>                     m_provider;
    std::shared_ptr<batch_decoder>                          m_decoder;
    std::shared_ptr<async_manager_source<fixed_buffer_map>> m_final_stage;
    int                                                     m_batch_size;
    BatchMode                                               m_batch_mode;
    int                                                     m_batch_count_value;
    size_t                                                  m_position{0};
    fixed_buffer_map*                                       m_output_buffer_ptr{nullptr};
    nlohmann::json                                          m_current_config;
    std::shared_ptr<web_app>                                m_debug_web_app;

    // Shows how bigger should be batch size than CPU thread count to not use extended pipeline which increase input size for decoder
    const float m_increase_input_size_coefficient = 1.5;
    // How many times we should increase input data size for decoder
    const int m_input_multiplier = 8;
};
