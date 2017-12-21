/*
 Copyright 2017 Intel(R) Nervana(TM)
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

#include <algorithm>

#include "loader_remote.hpp"
#include "remote_config.hpp"

using nlohmann::json;
using std::exception;
using std::runtime_error;
using std::map;
using std::shared_ptr;
using std::string;
using std::stringstream;
using std::vector;

using nervana::fixed_buffer_map;
using nervana::shape_type;
using nervana::shape_t;

namespace
{
    std::exception service_exception(const nervana::service_status& status, const string& action);
}

nervana::loader_remote::loader_remote(shared_ptr<service> client, const string& config)
    : m_service(client)
    , m_current_iter(*this, false)
    , m_end_iter(*this, true)
{
    m_config = json::parse(config);
    initialize();
}

nervana::loader_remote::loader_remote(shared_ptr<service> client, const nlohmann::json& config)
    : m_config(config)
    , m_service(client)
    , m_current_iter(*this, false)
    , m_end_iter(*this, true)
{
    initialize();
}

nervana::loader_remote::~loader_remote()
{
    if (!m_session_id.empty() && m_close_session)
    {
        close_session();
    }
}

void nervana::loader_remote::initialize()
{
    remote_config config{m_config["remote"]};
    m_session_id    = config.session_id;
    m_close_session = config.close_session;
    // If there is a session_id provided, we share already created session. Otherwise, we create a new one.
    if (m_session_id.empty())
    {
        create_session();
    }
    else
    {
        m_shared_session = true;
    }

    if (!config.debug_output_directory.empty())
    {
        m_output_saver =
            std::unique_ptr<output_saver>(new output_saver(config.debug_output_directory));
    }

    retrieve_names_and_shapes();
    retrieve_record_count();
    retrieve_batch_size();
    retrieve_batch_count();
}

void nervana::loader_remote::create_session()
{
    try
    {
        service_response<string> response = m_service->create_session(m_config.dump());
        response.status.assert_success();
        m_session_id = response.data;
    }
    catch (exception& ex)
    {
        throw runtime_error(string("cannot create session: ") + ex.what());
    }
}

void nervana::loader_remote::close_session()
{
    try
    {
        service_status status = m_service->close_session(m_session_id);
        status.assert_success();
    }
    catch (exception& ex)
    {
        throw runtime_error(string("cannot close session: ") + ex.what());
    }
}

const vector<string>& nervana::loader_remote::get_buffer_names() const
{
    if (m_names.empty())
    {
        std::transform(m_names_and_shapes.begin(),
                       m_names_and_shapes.end(),
                       std::back_inserter(m_names),
                       [](const std::pair<std::string, shape_type>& item) -> std::string {
                           return std::get<0>(item);
                       });
    }
    return m_names;
}

const shape_t& nervana::loader_remote::get_shape(const string& name) const
{
    auto found = std::find_if(
        m_names_and_shapes.begin(),
        m_names_and_shapes.end(),
        [&](const std::pair<std::string, shape_type>& item) { return std::get<0>(item) == name; });
    if (found != std::end(m_names_and_shapes))
    {
        return std::get<1>(*found).get_shape();
    }
    std::stringstream ss;
    ss << "key '" << name << "' not found";
    throw std::runtime_error(ss.str());
}

void nervana::loader_remote::retrieve_record_count()
{
    try
    {
        auto response = m_service->get_record_count(m_session_id);
        response.status.assert_success();
        m_record_count = response.data;
    }
    catch (exception& ex)
    {
        throw runtime_error(string("cannot retrieve record count: ") + ex.what());
    }
}

void nervana::loader_remote::retrieve_names_and_shapes()
{
    try
    {
        auto response  = m_service->get_names_and_shapes(m_session_id);
        m_record_count = -1;
        response.status.assert_success();
        m_names_and_shapes = response.data;
    }
    catch (exception& ex)
    {
        throw runtime_error(string("cannot retrieve names and shapes: ") + ex.what());
    }
}

void nervana::loader_remote::retrieve_batch_size()
{
    try
    {
        auto response = m_service->get_batch_size(m_session_id);
        m_batch_size  = -1;
        response.status.assert_success();
        m_batch_size = response.data;
    }
    catch (exception& ex)
    {
        throw runtime_error(string("cannot retrieve batch size: ") + ex.what());
    }
}

void nervana::loader_remote::retrieve_batch_count()
{
    try
    {
        auto response = m_service->get_batch_count(m_session_id);
        m_batch_count = -1;
        response.status.assert_success();
        m_batch_count = response.data;
    }
    catch (exception& ex)
    {
        throw runtime_error(string("cannot retrieve batch count: ") + ex.what());
    }
}

void nervana::loader_remote::retrieve_next_batch()
{
    service_response<next_response> response;
    try
    {
        response = m_service->get_next(m_session_id);
    }
    catch (exception& ex)
    {
        throw runtime_error(string("cannot retrieve next batch: ") + ex.what());
    }
    if (response.status.type != service_status_type::SUCCESS)
    {
        if (response.status.type == service_status_type::END_OF_DATASET)
        {
            m_position = m_batch_count;
        }
        else
        {
            throw service_exception(response.status, "cannot retrieve next batch");
        }
    }
    else if (m_output_saver)
    {
        m_output_saver->save(response.data.data.get());
    }

    const next_response& next = response.data;
    m_output_buffer_ptr       = next.data;
}

nervana::loader::iterator nervana::loader_remote::begin()
{
    reset();
    retrieve_next_batch();
    return m_current_iter;
}

void nervana::loader_remote::reset()
{
    // client sharing session cannot reset data
    if (m_shared_session)
        return;
    auto status = m_service->reset_session(m_session_id);
    if (!status.success())
    {
        throw service_exception(status, "cannot reset session");
    }
    m_batch_to_fetch = true;
    m_position       = 0;
}

nervana::loader_remote::iterator& nervana::loader_remote::get_current_iter()
{
    // for remote loader we dont fetch data in reset, because client starting session may not want to use data
    if (m_batch_to_fetch)
    {
        retrieve_next_batch();
        m_batch_to_fetch = false;
    }
    return m_current_iter;
}

void nervana::loader_remote::increment_position()
{
    retrieve_next_batch();
}

namespace
{
    std::exception service_exception(const nervana::service_status& status, const string& action)
    {
        stringstream ss;
        ss << action << ": service response failure ["
           << "status: " << to_string(status.type) << "; description: " << status.description
           << "]";
        throw std::runtime_error(ss.str());
    }
}
