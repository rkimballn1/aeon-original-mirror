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
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstdio>
#include <iostream>
#include <chrono>
#include <utility>
#include <algorithm>
#include <functional>
#include <future>
#include <map>
#include <type_traits>
#include "blocking_queue.hpp"
#include "log.hpp"

namespace nervana
{
    template <typename OUTPUT>
    class async_manager_source;
    template <typename INPUT, typename OUTPUT>
    class async_manager;
    class async_manager_info;

    enum class async_state
    {
        idle,
        wait_for_buffer,
        fetching_data,
        processing
    };
    extern std::vector<async_manager_info*> async_manager_status;
}

class nervana::async_manager_info
{
public:
    virtual ~async_manager_info() {}
    virtual async_state        get_state() const = 0;
    virtual const std::string& get_name() const  = 0;
};

template <typename T>
class sync_queues
{
public:
    std::unique_ptr<nervana::blocking_queue<std::shared_ptr<T>>> m_done_queue;
    std::unique_ptr<nervana::blocking_queue<std::shared_ptr<T>>> m_ready_queue;

//    size_t m_prefetch_size;

public:
    sync_queues(/*size_t prefetch_size*/)
        : m_done_queue{new nervana::blocking_queue<std::shared_ptr<T>>}
        , m_ready_queue{new nervana::blocking_queue<std::shared_ptr<T>>}
//        , m_prefetch_size{prefetch_size}
    { }
};

template <typename INPUT, typename OUTPUT, typename T>
class node
{
    size_t m_prefetch_size;

public:
    std::shared_ptr<sync_queues<INPUT>> m_input_queues;
    std::shared_ptr<sync_queues<OUTPUT>> m_output_queues;

    std::unique_ptr<T> m_worker;
    std::thread m_worker_thread;

public:
    using input_type = INPUT;
    using output_type = OUTPUT;

    template <typename ...ARGS>
    node(size_t prefetch_size, ARGS... args)
        : m_prefetch_size{prefetch_size}
        , m_input_queues{new sync_queues<INPUT>}
        , m_output_queues{new sync_queues<OUTPUT>}
        , m_worker{new T(args...)}
    {
        m_worker_thread = std::thread{&node::entry, this};   
    }

    void entry()
    {
        for (;;) 
        {
            for (size_t i = 0; i < m_prefetch_size; ++i) {
                std::shared_ptr<INPUT> in;
                std::shared_ptr<OUTPUT> out;

                m_input_queues->m_ready_queue->pop(in);
                m_output_queues->m_done_queue->pop(out);

//                m_worker->filler(in, out);
                m_worker->filler();

                m_input_queues->m_done_queue->push(in);
                m_output_queues->m_ready_queue->push(out);
            }
        }
    }

    ~node()
    {
        m_worker_thread.join();
    }
};

template <typename OUTPUT, typename T>
class start_node
{
    size_t m_prefetch_size;
    std::shared_ptr<sync_queues<OUTPUT>> m_output_queues;

    std::unique_ptr<T> m_worker;
    std::thread m_worker_thread;

public:
//    typename OUTPUT output_type;

    template <class... ARGS>
    start_node(size_t prefetch_size, ARGS... args)
        : m_prefetch_size{prefetch_size}
        , m_output_queues{new sync_queues<OUTPUT>}
        , m_worker{new T(args...)}
    { 
        m_worker_thread = std::thread{&start_node::entry, this};
    }

    void entry()
    {
        for (;;)
        {
            for (size_t i = 0; i < m_prefetch_size; ++i)
            {
                std::shared_ptr<OUTPUT> out;

                m_output_queues->m_done_queue->pop(out);
                m_worker->filler();
//                m_worker->filler(out);
                m_output_queues->m_ready_queue->push(out);
            }
        }
    }

    ~start_node()
    {
        m_worker_thread.join();
    }
};

template <typename INPUT, typename T>
class end_node
{
public:
    sync_queues<INPUT> m_input_queues;

    std::unique_ptr<T> m_worker;
    std::thread m_worker_thread;

    size_t m_prefetch_size;

public:
    using input_type = INPUT;

    template <typename ...ARGS>
    end_node(size_t prefetch_size, ARGS... args)
        : m_prefetch_size{prefetch_size}
        , m_input_queues{new sync_queues<INPUT>{m_prefetch_size}}
        , m_worker{new T{args...}}
    { 
        m_worker_thread = std::thread{&end_node::entry, this};
    }

    void entry()
    {
        for (;;)
        {
            std::shared_ptr<INPUT> in;

            m_input_queues.m_ready_queue->pop(in);
            m_worker->filler(in);
            m_input_queues.m_done_queue->push(in);
        }
    }

    ~end_node()
    {
        m_worker_thread.join();
    }
};

template<typename LHS, typename RHS>
void connect_nodes(LHS& lhs, RHS& rhs)
{
    static_assert(std::is_same<typename LHS::input_type, typename RHS::output_type>::value, "Different types for LHS and RHS");

    using connection_type = typename LHS::input_type;

    std::shared_ptr<sync_queues<connection_type>> q1{new sync_queues<connection_type>};

    rhs.m_output_queues->m_ready_queue.swap(q1->m_ready_queue);
    lhs.m_input_queues->m_done_queue.swap(q1->m_done_queue);   
}

template <typename OUTPUT>
class nervana::async_manager_source
{
public:
    async_manager_source() {}
    virtual ~async_manager_source() {}
    virtual OUTPUT* next()                      = 0;
    virtual size_t  record_count() const        = 0;
    virtual size_t  elements_per_record() const = 0;
    virtual void    reset()                     = 0;

    async_manager_source(const async_manager_source&) = default;
};

template <typename INPUT, typename OUTPUT>
class nervana::async_manager : public nervana::async_manager_source<OUTPUT>,
                               public async_manager_info
{
public:
    async_manager(async_manager_source<INPUT>* source, const std::string& name)
        : m_source(source)
        , m_state{async_state::idle}
        , m_name{name}
    {
        // Make the container pair?  Currently letting child handle it in filler()
        async_manager_status.push_back(this);
    }

    OUTPUT* next() override
    {
        // Special case for first time through
        OUTPUT* result = nullptr;
        if (m_first)
        {
            m_first = false;
            // Just run this one in blocking mode
            m_pending_result = std::async(
                std::launch::async, &nervana::async_manager<INPUT, OUTPUT>::filler, this);
        }
        result = m_pending_result.get();
        if (result != nullptr)
        {
            swap();

            // Now kick off this one in async
            m_pending_result = std::async(
                std::launch::async, &nervana::async_manager<INPUT, OUTPUT>::filler, this);
        }
        return result;
    }

    // do the work to fill up m_containers
    virtual OUTPUT* filler() = 0;

    virtual void reset() override
    {
        finalize();
        m_source->reset();
        initialize();
    }

    virtual ~async_manager() { finalize(); }
    virtual void initialize() { m_first = true; }
    void         finalize()
    {
        if (m_pending_result.valid())
        {
            m_pending_result.get();
        }
    }

    async_state        get_state() const override { return m_state; }
    const std::string& get_name() const override { return m_name; }
protected:
    async_manager(const async_manager&) = delete;
    void swap()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_index_done = m_index_pend;
        m_index_pend = m_index_pend == 1 ? 0 : 1;
    }

    OUTPUT*                      get_pending_buffer() { return &m_containers[m_index_pend]; }
    std::mutex                   m_mutex;
    OUTPUT                       m_containers[2];
    int                          m_index_pend{0};
    int                          m_index_done{0};
    std::future<OUTPUT*>         m_pending_result;
    bool                         m_first{true};
    async_manager_source<INPUT>* m_source;

    async_state m_state = async_state::idle;
    std::string m_name;
};
