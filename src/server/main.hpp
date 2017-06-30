#include <iostream>
#include <memory>
#include <cpprest/http_listener.h>

#include "json.hpp"
#include "loader.hpp"
#include "manifest_builder.hpp"
#include "gen_image.hpp"

class aeon_server
{
public:
    aeon_server(utility::string_t url);

    pplx::task<void> open();
    pplx::task<void> close();

private:
    void handle_get(web::http::http_request message);
    
    web::http::experimental::listener::http_listener m_listener;
};

std::shared_ptr<aeon_server> initialize(const utility::string_t& address);

void shutdown(std::shared_ptr<aeon_server> server);

class tpatejko
{
    std::shared_ptr<aeon_server> server;
public:
    tpatejko();
    ~tpatejko();
};
