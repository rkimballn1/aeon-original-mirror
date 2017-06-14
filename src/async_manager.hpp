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
#include "event.hpp"

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

enum class message
{
    RESET,
    TERMINATE,
    DATA_SENT,
    DATA_RECEIVED,
    END_OF_DATA
};

enum class node_state
{
    RUNNING,
    SUSPENDED,
    TERMINATED
};

template <typename T>
class sync_queues
{
public:
    std::shared_ptr<nervana::blocking_queue<message>> m_in_comm_queue;
    std::shared_ptr<nervana::blocking_queue<message>> m_out_comm_queue;
    std::shared_ptr<nervana::blocking_queue<T>> m_data_queue;

public:
    sync_queues(/*size_t prefetch_size*/)
//        : m_data_queue{new nervana::blocking_queue<T>}
    { }
};

template <typename INPUT, typename OUTPUT, typename T>
class node
{
    std::string m_name;
    node_state m_state;

public:
    std::shared_ptr<sync_queues<INPUT>> m_incoming_queues;
    std::shared_ptr<sync_queues<OUTPUT>> m_outgoing_queues;

    std::unique_ptr<T> m_worker;
    std::thread m_worker_thread;

public:
    using input_type = INPUT;
    using output_type = OUTPUT;

    template <typename ...ARGS>
    node(const std::string& name, ARGS... args)
        : m_name{name}
        , m_incoming_queues{new sync_queues<INPUT>}
        , m_outgoing_queues{new sync_queues<OUTPUT>}
        , m_worker{new T(args...)}
    {
    }

    void start()
    {
        m_state = node_state::RUNNING;
        m_worker_thread = std::thread{&node::entry, this};
    }

    void entry()
    {
        for (;;) 
        {
            while (m_state == node_state::RUNNING)
            {
                if (!m_incoming_queues->m_in_comm_queue->empty())
                {
                    message m;
                    m_incoming_queues->m_in_comm_queue->pop(m);

                    if (m == message::END_OF_DATA)
                    {
                        m_outgoing_queues->m_out_comm_queue->push(message::END_OF_DATA);
                        m_state = node_state::SUSPENDED;
                        INFO << m_name << " End of data";
                        break;
                    }
                }
                else if (!m_incoming_queues->m_data_queue->empty())
                {
                    INPUT in;

                    m_incoming_queues->m_data_queue->pop(in);
                    OUTPUT out = m_worker->fill(in);
                    m_outgoing_queues->m_data_queue->push(std::move(out));

                    INFO << m_name << " data received and sent";
                }
            }

            while (m_state == node_state::SUSPENDED)
            {
                if (!m_incoming_queues->m_in_comm_queue->empty())
                {
                    message m;
                    m_incoming_queues->m_in_comm_queue->pop(m);

                    if (m == message::RESET)
                    {
                        INFO << m_name << " Reset received. Sending reset";
                        m_outgoing_queues->m_out_comm_queue->push(message::RESET);

                    }
                    else if (m == message::TERMINATE)
                    {
                        m_outgoing_queues->m_out_comm_queue->push(message::TERMINATE);
                    }
                }

                if (!m_outgoing_queues->m_in_comm_queue->empty())
                {
                    INFO << m_name << " not empty";

                    message m;
                    m_outgoing_queues->m_in_comm_queue->pop(m);

                    if (m == message::RESET)
                    {
                        m_incoming_queues->m_out_comm_queue->push(message::RESET);
                        INFO << m_name << " Reset";
                        m_worker->reset();
                        m_state = node_state::RUNNING;
                        break;
                    }
                    else if (m == message::TERMINATE)
                    {
                        m_incoming_queues->m_out_comm_queue->push(message::TERMINATE);
                        m_state = node_state::TERMINATED;
                        return;
                    }
                }
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
    std::string m_name;

public:
    node_state m_state;

    std::shared_ptr<sync_queues<OUTPUT>> m_outgoing_queues;

    std::unique_ptr<T> m_worker;
    std::thread m_worker_thread;

public:
    using output_type = OUTPUT;
    
    template <class... ARGS>
    start_node(const std::string& name, ARGS... args)
        : m_name{name}
        , m_outgoing_queues{new sync_queues<OUTPUT>}
        , m_worker{new T(args...)}
    {
    }

    void start()
    {
        m_state = node_state::RUNNING;
        m_worker_thread = std::thread{&start_node::entry, this};
    }

    void entry()
    {
        for (;;)
        {
            while (m_state == node_state::RUNNING)
            {
                OUTPUT out = m_worker->fill();
            
                if (out.size() == 0)
                {
                    INFO << m_name << " Send END_OF_DATA";
                    m_outgoing_queues->m_out_comm_queue->push(message::END_OF_DATA);
                    m_state = node_state::SUSPENDED;
                    break;
                }
                else
                {
                    m_outgoing_queues->m_data_queue->push(std::move(out));
                }
            }

            while (m_state == node_state::SUSPENDED)
            {
                if (!m_outgoing_queues->m_in_comm_queue->empty())
                {
                    message m;
                    m_outgoing_queues->m_in_comm_queue->pop(m);

                    if (m == message::RESET)
                    {
                        INFO << m_name << " Do RESET";
                        m_worker->reset();
                        m_state = node_state::RUNNING;
                        break;
                    }
                    else if (m == message::TERMINATE)
                    {
                        m_state = node_state::TERMINATED;
                        return;
                    }
                }
            }
        }
    }

    ~start_node()
    {
        m_worker_thread.join();
    }
};

template <typename INPUT>
class end_node
{
    std::string m_name;
    node_state m_state;
public:
    std::shared_ptr<sync_queues<INPUT>> m_incoming_queues;
    std::thread m_worker_thread;

public:
    using input_type = INPUT;

    end_node(const std::string& name)
        : m_name{name}
        , m_incoming_queues{new sync_queues<INPUT>}
    { 
    }
    
    void start()
    {
        m_state = node_state::RUNNING;
        m_worker_thread = std::thread{&end_node::entry, this};
    }

    void entry()
    {
        for (;;)
        {
            while (m_state == node_state::RUNNING)
            {
                if (!m_incoming_queues->m_in_comm_queue->empty())
                {
                    message m;

                    m_incoming_queues->m_in_comm_queue->pop(m);

                    if (m == message::END_OF_DATA)
                    {
                        m_incoming_queues->m_out_comm_queue->push(message::RESET);
                    }
                }
            }
        }
    }

    INPUT get_datum()
    {
        INFO << m_name << " get datum";


        INPUT in;
        m_incoming_queues->m_data_queue->pop(in);
    
        return in;
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

    rhs.m_outgoing_queues->m_data_queue = std::make_shared<nervana::blocking_queue<connection_type>>();
    lhs.m_incoming_queues->m_data_queue = rhs.m_outgoing_queues->m_data_queue;

    rhs.m_outgoing_queues->m_in_comm_queue = std::make_shared<nervana::blocking_queue<message>>();
    lhs.m_incoming_queues->m_out_comm_queue = rhs.m_outgoing_queues->m_in_comm_queue;

    rhs.m_outgoing_queues->m_out_comm_queue = std::make_shared<nervana::blocking_queue<message>>();
    lhs.m_incoming_queues->m_in_comm_queue = rhs.m_outgoing_queues->m_out_comm_queue;
}

template <typename OUTPUT>
class nervana::async_manager_source
{
public:
    async_manager_source() {}
    virtual ~async_manager_source() {}
//    virtual OUTPUT* next()                      = 0;
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

/*
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
*/
    // do the work to fill up m_containers
    virtual OUTPUT fill(INPUT&) = 0;

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

    //OUTPUT*                      get_pending_buffer() { return &m_containers[m_index_pend]; }
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
