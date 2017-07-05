#include <chrono>

#include <cpprest/http_client.h>

using namespace web;
using namespace http;

static const size_t count = 10000;

void communicate(client::http_client& client)
{
    http_response response = client.request(methods::GET).get();

    if (response.status_code() != status_codes::OK)
    {
        throw std::runtime_error("Invalid response status");
    }
}

void communicate(const http::uri& uri)
{
    client::http_client client(http::uri_builder(uri).append_path(U("/aeon")).to_uri());
    http_response response = client.request(methods::GET).get();

    if (response.status_code() != status_codes::OK)
    {
        throw std::runtime_error("Invalid response status");
    }
}

int main()
{
    utility::string_t port = U("34568");
    utility::string_t address = U("http://127.0.0.1:");
    address.append(port);

    http::uri uri = http::uri(address);
    client::http_client client(http::uri_builder(uri).append_path(U("/aeon")).to_uri());

   
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < count; i++) {
        communicate(client);
    }
    auto stop = std::chrono::high_resolution_clock::now();

    auto dur = stop - start;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur);
    std::cout << "Duration:   Total: " << ms.count()             << " seconds." << std::endl <<
                 "          Average: " << (1.0*std::chrono::duration_cast<std::chrono::milliseconds>(ms).count())/count << " microseconds." << std::endl;

    return 0;
}
