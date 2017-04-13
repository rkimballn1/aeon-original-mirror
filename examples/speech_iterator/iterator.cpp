#include <iostream>
#include <fstream>

#include "aeon.hpp"


int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: speech_iterator <manifest_root> <manifest>" << std::endl;
        exit(1);
    }

    size_t batch_size         = 32;
    std::string manifest_root{argv[1]};
    std::string manifest{argv[2]};
    std::string alphabet      = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ ";


    nlohmann::json feats_config = {{"type", "audio"},
                                   {"sample_freq_hz", 16000},
                                   {"max_duration", "7.5 seconds"},
                                   {"frame_length", "0.025 seconds"},
                                   {"frame_stride", "0.01 seconds"},
                                   {"feature_type", "mfsc"},
                                   {"num_filters", 13}};

    nlohmann::json label_config = {{"type", "char_map"},
                                   {"max_length", 125},
                                   {"alphabet", alphabet},
                                   {"emit_length", true}};

    nlohmann::json config = {{"manifest_root", manifest_root},
                             {"manifest_filename", manifest},
                             {"batch_size", batch_size},
                             {"iteration_mode", "COUNT"},
                             {"iteration_mode_count", 10000},
                             {"etl", {feats_config, label_config}},
                             {"shuffle_enable", false},
                             {"shuffle_manifest", false},
                             {"random_seed", 0}
                             };

    auto train_set = nervana::loader{config};
    int iteration_index = 0;

    for (const nervana::fixed_buffer_map& x : train_set)
    {
        int transcript_length = *(int *)(x["length"]->get_item(0));
        std::cout << "INDEX: " << iteration_index << " " << x["audio"]->size() << " ";
        std::cout << "Transcript length: " << transcript_length << std::endl;
        iteration_index++;
    }
}
