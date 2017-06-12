#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>

#include "log.hpp"

namespace nervana {

template<typename T>
class blocking_queue 
{
public:
    blocking_queue()
    { }

    blocking_queue(const blocking_queue<T>&) = delete;
    blocking_queue<T>& operator=(const blocking_queue<T>&) = delete;

    void push(T&& t)
    {
        std::lock_guard<std::mutex> lk{m_mutex};
        m_queue.push(std::move(t));
        m_cond_var.notify_one();
    }

    bool try_pop(T& t)
    {
        std::lock_guard<std::mutex> lk{m_mutex};
        if (m_queue.empty())
            return false;

        t = std::move(m_queue.front());
        m_queue.pop();
        return true;
    }

    void pop(T& t)
    {
        std::unique_lock<std::mutex> lk{m_mutex};

        m_cond_var.wait(lk, [this]() -> bool { return !m_queue.empty(); });

        t = std::move(m_queue.front());
        m_queue.pop();
    }

    bool empty()
    {
        std::lock_guard<std::mutex> lk{m_mutex};
        return m_queue.empty();
    }

    size_t size()
    {
        std::lock_guard<std::mutex> lk{m_mutex};
        return m_queue.size();
    }

    bool is_size(size_t size)
    {
        std::lock_guard<std::mutex> lk{m_mutex};
        return (m_queue.size() == size);
    }

private:
    std::queue<T>           m_queue;
    mutable std::mutex      m_mutex;
    std::condition_variable m_cond_var;
};

}  // namespace nervana

