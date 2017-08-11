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

#include "service.hpp"
#include "../base64.hpp"

using nlohmann::json;
using std::exception;
using std::istringstream;
using std::ostringstream;
using std::invalid_argument;
using std::map;
using std::runtime_error;
using std::string;

using nervana::base64;
using nervana::names_and_shapes;
using nervana::service_status_type;
using nervana::service_response;
using nervana::service_status;
using nervana::service_status_type;

namespace
{
    const string version         = "v1";
    const string endpoint_prefix = "/api/" + version + "/dataset";

    string full_endpoint(const string& resource);

    string get_string_field(const json& input, const string& key, const service_status& status);
    int get_int_field(const json& input, const string& key, const service_status& status);
}

string nervana::to_string(service_status_type type)
{
    static map<service_status_type, string> status_map = {
        {service_status_type::SUCCESS, "SUCCESS"},
        {service_status_type::FAILURE, "FAILURE"},
        {service_status_type::END_OF_DATASET, "END_OF_DATASET"},
        {service_status_type::UNDEFINED, "UNDEFINED"}};
    return status_map[type];
}

service_status_type nervana::service_status_type_from_string(const string& input)
{
    static map<string, service_status_type> status_map = {
        {"SUCCESS", service_status_type::SUCCESS},
        {"FAILURE", service_status_type::FAILURE},
        {"END_OF_DATASET", service_status_type::END_OF_DATASET},
        {"UNDEFINED", service_status_type::UNDEFINED}};
    if (status_map.find(input) == status_map.end())
    {
        throw std::invalid_argument("undefined service_status_type: " + input);
    }
    return status_map[input];
}

nervana::service_status::service_status(const json& input)
{
    string type_str;
    try
    {
        type_str = input.at("type");
    }
    catch (const exception&)
    {
        throw invalid_argument("cannot parse service_status: no 'type' field provided");
    }

    try
    {
        type = service_status_type_from_string(type_str);
    }
    catch (const exception& ex)
    {
        throw invalid_argument(string("cannot parse service_status: ") + ex.what());
    }

    try
    {
        description = input.at("description");
    }
    catch (const exception&) // it's ok to not have description
    {
        description = "";
    }
}

string nervana::service_status::to_string() const
{
    static map<service_status_type, string> status_map = {
        {service_status_type::SUCCESS, "SUCCESS"},
        {service_status_type::END_OF_DATASET, "END_OF_DATASET"}};
    return status_map[type];
}

nervana::service_connector::service_connector(std::shared_ptr<http_connector> http)
    : m_http(http)
{
}

service_response<string> nervana::service_connector::create_session(const std::string& config)
{
    http_response response = m_http->post(full_endpoint(""), config);
    if (response.code != http::status_accepted && response.code != http::status_created)
    {
        throw runtime_error("wrong http status code " + std::to_string(response.code));
    }

    service_status status;
    json           json_response;
    extract_status_and_json(response.data, status, json_response);
    if (status.type != service_status_type::SUCCESS)
    {
        return service_response<string>(status, "");
    }

    string session_id = get_string_field(json_response, "id", status);
    if (session_id.empty())
    {
        throw std::runtime_error("service returned empty session id: " + status.to_string());
    }
    return service_response<string>(status, session_id);
}

service_response<nervana::next_response> nervana::service_connector::next(const string& id)
{
    http_response response = m_http->get(full_endpoint(id + "/next"));
    if (response.code != http::status_ok)
    {
        throw runtime_error("wrong http status code " + std::to_string(response.code));
    }
    service_status status;
    json           json_response;
    extract_status_and_json(response.data, status, json_response);
    if (status.type != service_status_type::SUCCESS)
    {
        return service_response<next_response>(status, next_response());
    }

    int position = get_int_field(json_response, "position", status);

    string encoded_buffer_map;
    try
    {
        encoded_buffer_map = json_response.at("fixed_buffer_map");
    }
    catch (const exception&)
    {
        throw runtime_error("no field 'fixed_buffer_map' in 'data' object of service response");
    }

    std::vector<char> serialized_buffer_map =
        base64::decode(encoded_buffer_map.c_str(), encoded_buffer_map.size());

    try
    {
        istringstream stream(string(serialized_buffer_map.begin(), serialized_buffer_map.end()));
        m_buffer_map.deserialize(stream);
    }
    catch (const exception& ex)
    {
        throw runtime_error(string("cannot deserialize fixed_buffer_map: ") + ex.what());
    }

    return service_response<next_response>(status, next_response(position, &m_buffer_map));
}

service_status nervana::service_connector::reset(const string& id)
{
    http_response response = m_http->get(full_endpoint(id + "/reset"));
    if (response.code != http::status_ok)
    {
        throw runtime_error("wrong http status code " + std::to_string(response.code));
    }
    service_status status;
    json           json_response;
    extract_status_and_json(response.data, status, json_response);
    return status;
}

service_response<names_and_shapes>
    nervana::service_connector::get_names_and_shapes(const string& id)
{
    http_response response = m_http->get(full_endpoint(id + "/names_and_shapes"));
    if (response.code != http::status_ok)
    {
        throw runtime_error("wrong http status code " + std::to_string(response.code));
    }
    service_status status;
    json           json_response;
    extract_status_and_json(response.data, status, json_response);
    if (status.type != service_status_type::SUCCESS)
    {
        return service_response<names_and_shapes>(status, names_and_shapes());
    }

    string serialized_nas;
    try
    {
        serialized_nas = json_response.at("names_and_shapes");
    }
    catch (const exception&)
    {
        throw runtime_error("no field 'names_and_shapes' in 'data' object of service response");
    }

    names_and_shapes nas;
    try
    {
        istringstream stream(serialized_nas);
        stream >> nas;
    }
    catch (const exception& ex)
    {
        throw runtime_error(string("cannot deserialize names_and_shapes: ") + ex.what());
    }
    return service_response<names_and_shapes>(status, nas);
}

service_response<int> nervana::service_connector::record_count(const string& id)
{
    http_response response = m_http->get(full_endpoint(id + "/record_count"));
    return handle_single_int_response(response, "record_count");
}

service_response<int> nervana::service_connector::batch_size(const string& id)
{
    http_response response = m_http->get(full_endpoint(id + "/batch_size"));
    return handle_single_int_response(response, "batch_size");
}

service_response<int> nervana::service_connector::batch_count(const string& id)
{
    http_response response = m_http->get(full_endpoint(id + "/batch_count"));
    return handle_single_int_response(response, "batch_count");
}

service_response<int>
    nervana::service_connector::handle_single_int_response(http_response response,
                                                           const string& field_name)
{
    if (response.code != http::status_ok)
    {
        throw runtime_error("wrong http status code " + std::to_string(response.code));
    }
    service_status status;
    json           json_response;
    extract_status_and_json(response.data, status, json_response);
    if (status.type != service_status_type::SUCCESS)
    {
        return service_response<int>(status, -1);
    }
    int field_value = get_int_field(json_response, field_name, status);
    return service_response<int>(status, field_value);
}

void nervana::service_connector::extract_status_and_json(const string&   input,
                                                         service_status& status,
                                                         json&           output_json)
{
    json input_json;
    try
    {
        input_json = json::parse(input);
    }
    catch (const exception& ex)
    {
        throw runtime_error(string("cannot parse json: ") + ex.what());
    }

    json status_json;
    try
    {
        status_json = input_json.at("status");
    }
    catch (const exception&)
    {
        throw runtime_error("service response does not contain status field");
    }
    status = service_status(status_json);

    try
    {
        output_json = input_json.at("data");
    }
    catch (const exception&)
    {
        // no data was provided
        output_json = json({});
    }
}

namespace
{
    string full_endpoint(const string& resource)
    {
        return nervana::http::merge_http_paths(endpoint_prefix, resource);
    }

    string get_string_field(const json& input, const string& key, const service_status& status)
    {
        try
        {
            return input.at(key);
        }
        catch (const std::exception&)
        {
            throw std::runtime_error("response body does not contain string field '" + key + "': " +
                                     status.to_string());
        }
    }

    int get_int_field(const json& input, const string& key, const service_status& status)
    {
        string field;
        try
        {
            field = input.at(key);
        }
        catch (const std::exception&)
        {
            throw std::runtime_error("response body does not contain number field '" + key + "': " +
                                     status.to_string());
        }
        try
        {
            return std::stoi(field);
        }
        catch (const std::exception& ex)
        {
            throw std::runtime_error("cannot convert field '" + key + "' to integer: " + ex.what() +
                                     "; status: " + status.to_string());
        }
    }
}
