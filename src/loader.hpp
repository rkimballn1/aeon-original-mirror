/*
 Copyright 2016 Nervana Systems Inc.
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
    uint16_t                    web_server_port = 0;
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

class nervana::loader
{
public:
    enum class BatchMode
    {
        INFINITE,
        ONCE,
        COUNT
    };

    loader(const std::string&);
    loader(nlohmann::json&);

    ~loader();

    const std::vector<std::string>& get_buffer_names() const;
    const std::map<std::string, shape_type>& get_names_and_shapes() const;
    const shape_t& get_shape(const std::string& name) const;

    int record_count() { return m_manifest->record_count(); }
    int batch_size() { return m_batch_size; }
    // member typedefs provided through inheriting from std::iterator
    class iterator : public std::iterator<std::input_iterator_tag, // iterator_category
                                          fixed_buffer_map         // value_type
                                          // long,                     // difference_type
                                          // const fixed_buffer_map*,  // pointer
                                          // fixed_buffer_map          // reference
                                          >
    {
        friend class loader;

    public:
        explicit iterator(loader& ld, bool is_end);
        iterator(const iterator&);
        ~iterator() {}
        iterator& operator++(); // {num = TO >= FROM ? num + 1: num - 1; return *this;}
        iterator& operator++(int);
        bool operator==(const iterator& other) const; // {return num == other.num;}
        bool operator!=(const iterator& other) const; // {return !(*this == other);}
        const fixed_buffer_map& operator*() const;    // {return num;}
        const size_t& position() const { return m_current_loader.m_position; }
        bool          positional_end() const;

    private:
        iterator() = delete;

        loader&    m_current_loader;
        const bool m_is_end;
        fixed_buffer_map m_empty_buffer;
    };

    // Note that these are returning COPIES
    iterator begin()
    {
        reset();
        return m_current_iter;
    }

    iterator end() { return m_end_iter; }
    // These are returning references
    iterator& get_current_iter() { return m_current_iter; }
    iterator& get_end_iter() { return m_end_iter; }
    void      reset()
    {
        m_final_stage->reset();
        m_output_buffer_ptr = m_final_stage->next();
        m_position          = 0;
    }

    nlohmann::json get_current_config() const { return m_current_config; }
private:
    loader() = delete;
    void initialize(nlohmann::json& config_json);
    void increment_position();

    friend class nervana::loader::iterator;

    iterator                            m_current_iter;
    iterator                            m_end_iter;
    std::shared_ptr<manifest_file>      m_manifest;
    std::shared_ptr<block_loader_file>  m_block_loader;
    std::shared_ptr<block_manager>      m_block_manager;
    std::shared_ptr<batch_iterator>     m_batch_iterator;
    std::shared_ptr<provider_interface> m_provider;
    std::shared_ptr<batch_decoder>      m_decoder;
    std::shared_ptr<async_manager_source<fixed_buffer_map>> m_final_stage;
    int                                 m_batch_size;
    BatchMode                           m_batch_mode;
    size_t                              m_batch_count_value;
    size_t                              m_position{0};
    fixed_buffer_map*                   m_output_buffer_ptr{nullptr};
    nlohmann::json                      m_current_config;
    std::shared_ptr<web_app>            m_debug_web_app;
    
    // Shows how bigger should be batch size than CPU thread count to not use extended pipeline which increase input size for decoder
    const float                         m_increase_input_size_coefficient = 1.5; 
    // How many times we should increase input data size for decoder
    const int                           m_input_multiplier = 8;
};
