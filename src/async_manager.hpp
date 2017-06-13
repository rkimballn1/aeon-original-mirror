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
#include "log.hpp"
#include "blocking_queue.h"

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
    virtual void    suspend_output()              {}

    async_manager_source(const async_manager_source&) = default;
};


template <typename INPUT, typename OUTPUT>
class nervana::async_manager : public virtual nervana::async_manager_source<OUTPUT>,
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
    virtual ~async_manager() { finalize(); }
        
    void run_filler()
    {
      for(;;)
      {
        qinput.pop(m_pending_buffer);
        if (!m_active_thread)
            return;
        if (m_pending_buffer == nullptr)
        {
            qoutput.push(nullptr);
            return;
        }
        OUTPUT* buff = filler();
        if (!m_active_thread)
            return;
        qoutput.push(buff);
      }
    }
    
    OUTPUT* next() override
    {
        if (!m_active_thread) 
            initialize();
    
        OUTPUT* output_buffer;
        if (!m_bfirst_next)
        {
            qoutput.pop(output_buffer);
            if (output_buffer == nullptr)
                return nullptr; 
            qinput.push(output_buffer);
        }
        m_bfirst_next = false;

        qoutput.top(output_buffer);
        return output_buffer;
    }
 
    // do the work to fill up m_containers
    virtual OUTPUT* filler() = 0;
 
    virtual void reset() override
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (m_active_thread)
        {
            m_active_thread = false;
            qinput.clear();
            qinput.push(nullptr);
            m_source->suspend_output();
            fill_thread->join();
        }
        m_source->reset();
    }


    virtual void initialize() 
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (!m_active_thread)
        {
            m_active_thread = true;
            m_bfirst_next = true;
            qinput.clear();
            qoutput.clear();
            qinput.push(&m_containers[0]);
            qinput.push(&m_containers[1]);
            fill_thread.reset(new std::thread(&async_manager::run_filler, this));
        }
    }
   
    void finalize()
    {
      reset();
    }
 
    virtual void suspend_output() override
    {
        qoutput.clear();
        qoutput.push(nullptr);
    }

    async_state        get_state() const override { return m_state; }
    const std::string& get_name() const override { return m_name; }
protected:
    async_manager(const async_manager&) = delete;

    OUTPUT* get_pending_buffer()
    {
      if (m_active_thread)
        return m_pending_buffer;
      else 
        return &m_containers[0];
    }
    OUTPUT                       m_containers[2];
    OUTPUT*                      m_pending_buffer;
    async_manager_source<INPUT>* m_source;

    async_state m_state = async_state::idle;
    std::string m_name;
    
    // ///////////////////////////////////////////////////////////////////
    BlockingQueue<OUTPUT*>   qinput;
    BlockingQueue<OUTPUT*>  qoutput;
    std::shared_ptr<std::thread> fill_thread;
    volatile bool       m_bfirst_next{true};
    volatile bool       m_active_thread{false};
    std::mutex          m_mutex;
};

