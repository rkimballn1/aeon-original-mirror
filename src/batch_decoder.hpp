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

#include <thread>
#include <pthread.h>
#include <atomic>
#include <mutex>
#include <exception>

#include "async_manager.hpp"
#include "buffer_batch.hpp"
#include "provider_interface.hpp"
#include "provider_factory.hpp"
#include "event.hpp"

namespace nervana
{
    class batch_decoder;
    class decode_thread_info;
    template <typename T, void(T::*process_func)(int index)>
    class thread_pool;
    class batch_iterator;
}

template <typename T, void(T::*process_func)(int index)>
class nervana::thread_pool
{
public:
    thread_pool(T* worker, int thread_count, int task_count)
        :m_worker(worker)
        ,m_task_count(task_count)
    {
        int nthreads;

        if (thread_count == 0)  // automatically determine number of threads
        {
            // we don't use all threads, some of them we leave for other pipeline objects and system
            nthreads = std::thread::hardware_concurrency() - 
                       std::min(m_max_count_of_free_threads, 
                                static_cast<int>(std::thread::hardware_concurrency()/m_free_threads_ratio));
            nthreads = std::min(nthreads, m_task_count);
        }
        else
        {
            // don't return more threads than we can get
            nthreads = std::min(static_cast<int>(std::thread::hardware_concurrency()), thread_count);
        }
   
        pthread_barrier_init(&m_br_wake, NULL, nthreads + 1);
        pthread_barrier_init(&m_br_endtasks, NULL, nthreads + 1);

        if (nthreads == m_task_count)
        {
            for (int i = 0; i < nthreads; i++)
                m_threads.emplace_back(&thread_pool::process<false>, this, i);
        }
        else
        {
            for (int i = 0; i < nthreads; i++)
                m_threads.emplace_back(&thread_pool::process<true>, this, i);
        }
    }
    
    ~thread_pool()
    {
        m_thread_pool_stop = true;
        pthread_barrier_wait(&m_br_wake);
        for(auto &thread : m_threads)
            thread.join();
        pthread_barrier_destroy(&m_br_wake);
        pthread_barrier_destroy(&m_br_endtasks);
    }
    
    void run()
    {
        m_current_task_id = 0;
        m_pool_exception = nullptr;
        pthread_barrier_wait(&m_br_wake);
        pthread_barrier_wait(&m_br_endtasks);
        if (m_pool_exception)
            std::rethrow_exception(m_pool_exception);
    }
private:
    const int         m_max_count_of_free_threads = 2;
    const int         m_free_threads_ratio = 8;
    T*                m_worker;
    int               m_task_count;
    pthread_barrier_t m_br_wake;
    pthread_barrier_t m_br_endtasks;
    volatile bool     m_thread_pool_stop = false;
    std::vector<std::thread> m_threads;
    std::atomic<size_t> m_current_task_id;
    std::exception_ptr  m_pool_exception;
    std::mutex          m_mutex;
    
    template<bool dynamic_task_scheduling>
    void process(int thread_id)
    {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(thread_id, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset); 

        for(;;)
        {
            pthread_barrier_wait(&m_br_wake);

            if (m_thread_pool_stop) return;
            
            try
            {
                if (!dynamic_task_scheduling)
                {
                    (m_worker->*process_func)(thread_id);
                }
                else
                {
                    for(;;)
                    {
                        const size_t next_task_id = m_current_task_id.fetch_add(1);
                        if (next_task_id >= m_task_count) break;
                        (m_worker->*process_func)(next_task_id);
                    }
                }
            }
            catch(...)
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (!m_pool_exception)
                   m_pool_exception = std::current_exception();
            }

            pthread_barrier_wait(&m_br_endtasks);
        }
    }
};


class nervana::batch_decoder : public async_manager<encoded_record_list, fixed_buffer_map>
{
public:
    batch_decoder(batch_iterator*                            b_itor,
                  size_t                                     batch_size,
                  uint32_t                                   thread_count,
                  bool                                       pinned,
                  const std::shared_ptr<provider_interface>& prov);

    virtual ~batch_decoder();

    virtual size_t            record_count() const override { return m_batch_size; }
    virtual size_t            elements_per_record() const override { return m_number_elements_out; }
    virtual fixed_buffer_map* filler() override;

    void register_info_handler(std::function<void(const fixed_buffer_map*)>& f)
    {
        m_info_handler = f;
    }
    
    void process(const int index)
    {
        m_provider->provide(index, *m_inputs, *m_outputs);
    }

private:
    size_t m_batch_size;
    size_t m_number_elements_in;
    size_t m_number_elements_out;
    std::shared_ptr<provider_interface> m_provider;
    encoded_record_list*                m_inputs {nullptr};
    fixed_buffer_map*                   m_outputs{nullptr};
    thread_pool<batch_decoder, &batch_decoder::process> m_thread_pool;
    std::function<void(const fixed_buffer_map*)> m_info_handler;
};
