#include <iostream>
#include <fstream>

#include "aeon.hpp"
#include "file_util.hpp"

// To run this example, firstly run server with command:
// `cd server && ./aeon-server --address=http://127.0.0.1 --port 34568`

using nlohmann::json;
using std::cout;
using std::endl;
using std::shared_ptr;
using std::string;

using nervana::loader;
using nervana::loader_factory;
using nervana::manifest_file;

const string address = "127.0.0.1";
const int    port    = 34568;

string generate_manifest_file(const string& manifest_root, size_t record_count)
{
    string        manifest_name     = "manifest.txt";
    const char*   image_files[]     = {"flowers.jpg", "img_2112_70.jpg"};
    string        manifest_fullpath = manifest_root + "/" + manifest_name;
    std::ofstream f(manifest_fullpath);
    if (f)
    {
        f << manifest_file::get_metadata_char();
        f << manifest_file::get_file_type_id();
        f << manifest_file::get_delimiter();
        f << manifest_file::get_string_type_id();
        f << "\n";
        for (size_t i = 0; i < record_count; i++)
        {
            f << image_files[i % 2];
            f << manifest_file::get_delimiter();
            f << std::to_string(i % 2);
            f << "\n";
        }
    }
    return manifest_name;
}

int main(int argc, char** argv)
{
    int    height        = 32;
    int    width         = 32;
    size_t batch_size    = 4;
    string manifest_root = "server";
    string manifest      = generate_manifest_file(manifest_root, 20);

    json image_config = {
        {"type", "image"}, {"height", height}, {"width", width}, {"channel_major", false}};
    json label_config = {{"type", "label"}, {"binary", false}};
    json aug_config   = {{{"type", "image"}, {"flip_enable", true}}};
    json config       = {{"manifest_root", "./"},
                   {"manifest_filename", manifest},
                   {"batch_size", batch_size},
                   {"iteration_mode", "ONCE"},
                   {"etl", {image_config, label_config}},
                   {"augmentation", aug_config},
                   {"server", {{"address", address}, {"port", port}}}};

    // initialize loader object
    loader_factory     factory;
    shared_ptr<loader> train_set = factory.get_loader(config);

    // retrieve dataset info
    cout << "batch size: " << train_set->batch_size() << endl;
    cout << "batch count: " << train_set->batch_count() << endl;
    cout << "record count: " << train_set->record_count() << endl;

    // iterate through all data
    int batch_no = 0;
    for (const auto& batch : *train_set)
    {
        cout << "\tbatch " << batch_no << " [number of elements: " << batch.size() << "]" << endl;
        batch_no++;
    }
}
