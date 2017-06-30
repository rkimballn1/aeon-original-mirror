#include <iostream>
#include <memory>
#include <cpprest/http_listener.h>

#include "main.hpp"
#include "json.hpp"
#include "loader.hpp"
#include "manifest_builder.hpp"
#include "gen_image.hpp"

using namespace web::http;

pplx::task<void> aeon_server::open()
{
    return m_listener.open();
}

pplx::task<void> aeon_server::close()
{
    return m_listener.close();
}

aeon_server::aeon_server(utility::string_t url)
    : m_listener(url)
{
    m_listener.support(methods::GET,
                       std::bind(&aeon_server::handle_get, this, std::placeholders::_1));
}

void aeon_server::handle_get(http_request message)
{
    try
    {
        message.reply(status_codes::OK);
    }
    catch (std::exception& e)
    {
        throw e;
    }
}

std::shared_ptr<aeon_server> initialize(const utility::string_t& address)
{
    uri_builder uri(address);
    uri.append_path(U("aeon"));

    auto                         addr   = uri.to_uri().to_string();
    std::shared_ptr<aeon_server> server = std::make_shared<aeon_server>(addr);
    server->open().wait();

    return server;
}

void shutdown(std::shared_ptr<aeon_server> server)
{
    server->close().wait();
}

tpatejko::tpatejko()
{
    utility::string_t port = U("34568");

    utility::string_t address = U("http://127.0.0.1:");
    address.append(port);

    server = initialize(address);
}
tpatejko::~tpatejko()
{
    shutdown(server);
}
