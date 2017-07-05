#include <chrono>

#include <cpprest/http_client.h>

using namespace web;
using namespace http;

#define CAT(a, ...) PRIMITIVE_CAT(a, __VA_ARGS__)
#define PRIMITIVE_CAT(a, ...) a ## __VA_ARGS__

#define COMM_FUNC CAT(communicate_, MODE)

static const size_t count = 10000;

void communicate_par(client::http_client& client)
{
    std::vector<pplx::task<http_response>> pending_responses;
    pending_responses.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        pending_responses[i] = client.request(methods::GET);
    }

    for (auto pr : pending_responses) {
        http_response r = pr.get();

        if (r.status_code() != status_codes::OK)
            throw std::runtime_error("Invalid response status");
    }
}

void communicate_seq(client::http_client& client)
{
    for (size_t i = 0; i < count; ++i) {
        http_response response = client.request(methods::GET).get();

        if (response.status_code() != status_codes::OK)
        {
            throw std::runtime_error("Invalid response status");
        }
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
    COMM_FUNC(client);
    auto stop = std::chrono::high_resolution_clock::now();

    auto dur = stop - start;
    std::cout << "Duration:   Total: " << std::chrono::duration_cast<std::chrono::seconds>(dur).count() << " seconds." << std::endl <<
                 "          Average: " << (1.0*std::chrono::duration_cast<std::chrono::microseconds>(dur).count())/count << " microseconds." << std::endl;

    return 0;
}
