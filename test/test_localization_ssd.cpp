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

#include <vector>
#include <string>
#include <sstream>
#include <random>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "gtest/gtest.h"
#include "gen_image.hpp"
#include "json.hpp"

#define private public
#define protected public

#include "file_util.hpp"
#include "etl_localization_ssd.hpp"
#include "json.hpp"
#include "provider_factory.hpp"
#include "log.hpp"
#include "loader.hpp"
#include "test_localization.hpp"
#include "helpers.hpp"

using namespace std;
using namespace nervana;
using namespace nervana::localization::ssd;

using bbox = boundingbox::box;

TEST(localization_ssd, provider)
{
    int height = 375;
    int width  = 500;

    nlohmann::json js_image = {
        {"type", "image"}, {"height", height}, {"width", width}, {"channel_major", false}};
    nlohmann::json js_local = {{"type", "localization_ssd"},
                               {"height", height},
                               {"width", width},
                               {"max_gt_boxes", 64},
                               {"class_names", {"bicycle", "person"}},
                               {"gt_boxes_normalized", false}};
    nlohmann::json augmentation = {{{"type", "image"}, {"flip_enable", true}}};
    nlohmann::json js           = {{"etl", {js_local, js_image}}, {"augmentation", augmentation}};

    shared_ptr<provider_interface> media   = provider_factory::create(js);
    auto                           oshapes = media->get_output_shapes();
    ASSERT_NE(nullptr, media);
    ASSERT_EQ(6, oshapes.size());

    string       target = file_util::read_file_to_string(CURDIR "/test_data/006637.json");
    vector<char> target_data;
    target_data.insert(target_data.begin(), target.begin(), target.end());
    // Image size is from the 006637.json target data file
    cv::Mat image(375, 500, CV_8UC3);
    image = cv::Scalar(50, 100, 200);
    vector<uint8_t> image_data;
    vector<char>    image_cdata;
    cv::imencode(".png", image, image_data);
    for (auto c : image_data)
    {
        image_cdata.push_back(c);
    };

    encoded_record_list in_buf;
    encoded_record      record;
    record.add_element(target_data);
    record.add_element(image_cdata);
    in_buf.add_record(record);

    fixed_buffer_map  out_buf(oshapes, 1);
    const shape_type& image_shape = media->get_output_shape("image");

    media->provide(0, in_buf, out_buf);

    int     output_width  = image_shape.get_shape()[0];
    int     output_height = image_shape.get_shape()[1];
    int     channels      = image_shape.get_shape()[2];
    cv::Mat result(output_height, output_width, CV_8UC(channels), out_buf["image"]->get_item(0));
    //cv::imwrite("localization_provider_source.png", image);
    //cv::imwrite("localization_provider.png", result);

    uint8_t* data = result.data;
    for (int row = 0; row < result.rows; row++)
    {
        for (int col = 0; col < result.cols; col++)
        {
            ASSERT_EQ(50, data[0]) << "row=" << row << ", col=" << col;
            ASSERT_EQ(100, data[1]) << "row=" << row << ", col=" << col;
            ASSERT_EQ(200, data[2]) << "row=" << row << ", col=" << col;
            data += 3;
        }
    }
}

TEST(localization_ssd, extract_gt_boxes_unnormalized)
{
    int input_height  = 100;
    int input_width   = 100;
    int output_height = 200;
    int output_width  = 300;

    nlohmann::json js_local = {{"height", output_height},
                               {"width", output_width},
                               {"max_gt_boxes", 64},
                               {"class_names", {"bicycle", "person"}},
                               {"gt_boxes_normalized", false}};

    cv::Rect r0   = cv::Rect(0, 0, 10, 15);
    cv::Rect r1   = cv::Rect(10, 10, 12, 13);
    bbox     r2   = bbox(0.1f, 0.2f, 0.3f, 0.4f, true);
    auto     list = {create_box(r0, "bicycle"),
                 create_box(r1, "person"),
                 create_box_with_normalized_field(r2, "person")};
    auto   j      = create_metadata(list, input_width, input_height);
    string buffer = j.dump();

    localization::ssd::config    cfg{js_local};
    localization::ssd::extractor ext{cfg};
    auto                         decoded = ext.extract(&buffer[0], buffer.size());

    ASSERT_NE(nullptr, decoded);

    ASSERT_EQ(decoded->input_image_size, cv::Size2i(input_width, input_height));
    ASSERT_EQ(decoded->output_image_size, cv::Size2i(output_width, output_height));

    ASSERT_EQ(decoded->boxes().size(), 3);

    bbox expected  = bbox(0, 0, 9, 14, false);
    expected.label = 0;
    EXPECT_EQ(decoded->boxes()[0].rect(), expected.rect());

    expected       = bbox(10, 10, 21, 22, false);
    expected.label = 1;
    EXPECT_EQ(decoded->boxes()[1], expected);

    expected       = bbox(10, 20, 29, 39, false);
    expected.label = 1;
    EXPECT_EQ(decoded->boxes()[2], expected);
}

TEST(localization_ssd, extract_gt_boxes_normalized)
{
    int height = 100;
    int width  = 100;

    nlohmann::json js_local = {{"height", height},
                               {"width", width},
                               {"max_gt_boxes", 64},
                               {"class_names", {"bicycle", "person"}},
                               {"gt_boxes_normalized", true}};

    bbox r0   = bbox(0.0f, 0.0f, 1.0, 1.0, false);
    bbox r1   = bbox(0.2f, 0.4f, 0.3f, 0.5f, true);
    bbox r2   = bbox(0.0f, 0.0f, 0.0f, 0.0f, true);
    auto list = {create_box_with_normalized_field(r0, "bicycle"),
                 create_box_with_normalized_field(r1, "person"),
                 create_box_with_normalized_field(r2, "bicycle")};
    auto   j      = create_metadata(list, width, height);
    string buffer = j.dump();

    localization::ssd::config    cfg{js_local};
    localization::ssd::extractor ext{cfg};
    auto                         decoded = ext.extract(&buffer[0], buffer.size());

    ASSERT_NE(nullptr, decoded);
    ASSERT_EQ(decoded->boxes().size(), 3);

    bbox expected  = bbox(0, 0, 99, 99, false);
    expected.label = 0;
    EXPECT_EQ(decoded->boxes()[0], expected);

    expected       = bbox(20, 40, 29, 49, false);
    expected.label = 1;
    EXPECT_EQ(decoded->boxes()[1], expected);

    expected       = bbox(0, 0, -1, -1, false);
    expected.label = 0;
    EXPECT_EQ(decoded->boxes()[2], expected);
}

TEST(localization_ssd, extract_gt_boxes_normalized_with_wrongly_normalized_box)
{
    int height = 100;
    int width  = 100;

    nlohmann::json js_local = {{"height", height},
                               {"width", width},
                               {"max_gt_boxes", 64},
                               {"class_names", {"bicycle", "person"}},
                               {"gt_boxes_normalized", true}};

    bbox   r0     = bbox(0.2f, 0.4f, 0.3f, 1.5f, false);
    auto   list   = {create_box(r0, "bicycle")};
    auto   j      = create_metadata(list, width, height);
    string buffer = j.dump();

    localization::ssd::config    cfg{js_local};
    localization::ssd::extractor ext{cfg};
    EXPECT_THROW(ext.extract(&buffer[0], buffer.size()), std::invalid_argument);
}

TEST(localization_ssd, gt_boxes_normalized_with_normalized_field_false)
{
    int height = 100;
    int width  = 100;

    nlohmann::json js_local = {{"height", height},
                               {"width", width},
                               {"max_gt_boxes", 64},
                               {"class_names", {"bicycle", "person"}},
                               {"gt_boxes_normalized", true}};

    bbox   r0     = bbox(1, 2, 3, 4, false);
    auto   list   = {create_box_with_normalized_field(r0, "bicycle")};
    auto   j      = create_metadata(list, width, height);
    string buffer = j.dump();

    localization::ssd::config    cfg{js_local};
    localization::ssd::extractor ext{cfg};
    EXPECT_THROW(ext.extract(&buffer[0], buffer.size()), std::invalid_argument);
}

TEST(localization_ssd, transform)
{
    //just check that transformations are executed on bboxes (flip here) and they are eventually normalized
    int input_width   = 100;
    int input_height  = 100;
    int output_width  = 200;
    int output_height = 300;

    nlohmann::json js_local = {{"height", output_height},
                               {"width", output_width},
                               {"max_gt_boxes", 64},
                               {"class_names", {"bicycle", "person"}},
                               {"gt_boxes_normalized", true}};

    auto decoded               = make_shared<localization::ssd::decoded>();
    decoded->input_image_size  = cv::Size2i(input_width, input_height);
    decoded->output_image_size = cv::Size2i(output_width, output_height);
    decoded->m_boxes           = vector<bbox>{
        bbox(0, 0, 99, 99, false), bbox(20, 40, 29, 49, false), bbox(10, 20, 29, 39, false)};
    auto expected = vector<bbox>{bbox(0.0f, 0.0f, 1.0, 1.0, true),
                                 bbox(0.7f, 0.4f, 0.8f, 0.5f, true),
                                 bbox(0.7f, 0.2f, 0.9f, 0.4f, true)};
    augment::image::param_factory      factory({{"type", "image"}, {"crop_enable", false}});
    shared_ptr<augment::image::params> augmentation_params = factory.make_ssd_params(
        input_width, input_height, output_width, output_height, vector<bbox>());
    augmentation_params->flip = true;

    localization::ssd::config      cfg{js_local};
    localization::ssd::transformer trans{};
    trans.transform(augmentation_params, decoded);

    EXPECT_EQ(decoded->input_image_size, cv::Size2i(input_width, input_height));
    EXPECT_EQ(decoded->output_image_size, cv::Size2i(output_width, output_height));

    for (int i = 0; i < decoded->gt_boxes.size(); i++)
    {
        std::stringstream ss;
        ss << decoded->gt_boxes[i] << " is not equal to " << expected[i];
        EXPECT_EQ(decoded->gt_boxes[i], expected[i]) << ss.str();
    }
}

TEST(localization_ssd, loader)
{
    int height       = 100;
    int width        = 100;
    int max_gt_boxes = 64;

    nlohmann::json js_local = {{"height", height},
                               {"width", width},
                               {"max_gt_boxes", max_gt_boxes},
                               {"class_names", {"bicycle", "person", "cat"}},
                               {"gt_boxes_normalized", true}};

    bbox   r0     = bbox(1, 2, 3, 4, false);
    auto   list   = {create_box_with_normalized_field(r0, "bicycle")};
    auto   j      = create_metadata(list, width, height);
    string buffer = j.dump();

    localization::ssd::config    cfg{js_local};
    localization::ssd::extractor ext{cfg};
    EXPECT_THROW(ext.extract(&buffer[0], buffer.size()), std::invalid_argument);

    auto decoded               = make_shared<localization::ssd::decoded>();
    decoded->output_image_size = cv::Size2i(width, height);
    decoded->gt_boxes          = vector<bbox>{bbox(0.0, 0.0, 1.0, 1.0, true),
                                     bbox(0.1, 0.2, 0.3, 0.4, true),
                                     bbox(0.44444, 0.55555, 1.0, 0.99999, true)};
    decoded->gt_boxes[1].difficult = true;
    decoded->gt_boxes[1].label     = 1;
    decoded->gt_boxes[2].difficult = true;
    decoded->gt_boxes[2].label     = 2;

    auto          shape        = unique_ptr<int32_t>(new int32_t[2]);
    auto          gt_boxes     = unique_ptr<float>(new float[max_gt_boxes * 4]);
    auto          num_gt_boxes = unique_ptr<int32_t>(new int32_t[1]);
    auto          gt_classes   = unique_ptr<int32_t>(new int32_t[max_gt_boxes]);
    auto          gt_difficult = unique_ptr<int32_t>(new int32_t[max_gt_boxes]);
    vector<void*> buf_list;
    buf_list.push_back(static_cast<void*>(shape.get()));
    buf_list.push_back(static_cast<void*>(gt_boxes.get()));
    buf_list.push_back(static_cast<void*>(num_gt_boxes.get()));
    buf_list.push_back(static_cast<void*>(gt_classes.get()));
    buf_list.push_back(static_cast<void*>(gt_difficult.get()));

    localization::ssd::loader loader{cfg};
    loader.load(buf_list, decoded);

    ASSERT_EQ(shape.get()[0], width);
    ASSERT_EQ(shape.get()[1], height);
    ASSERT_EQ(*num_gt_boxes, decoded->gt_boxes.size());
    auto boxes_ptr     = gt_boxes.get();
    auto classes_ptr   = gt_classes.get();
    auto difficult_ptr = gt_difficult.get();
    for (int i = 0; i < decoded->gt_boxes.size(); i++)
    {
        bbox box = decoded->gt_boxes[i];
        EXPECT_FLOAT_EQ(*boxes_ptr++, box.xmin());
        EXPECT_FLOAT_EQ(*boxes_ptr++, box.ymin());
        EXPECT_FLOAT_EQ(*boxes_ptr++, box.xmax());
        EXPECT_FLOAT_EQ(*boxes_ptr++, box.ymax());
        EXPECT_EQ(*classes_ptr++, box.label);
        EXPECT_EQ(*difficult_ptr++, box.difficult);
    }
    for (int i = decoded->gt_boxes.size(); i < max_gt_boxes; i++)
    {
        EXPECT_FLOAT_EQ(*boxes_ptr++, 0);
        EXPECT_FLOAT_EQ(*boxes_ptr++, 0);
        EXPECT_FLOAT_EQ(*boxes_ptr++, 0);
        EXPECT_FLOAT_EQ(*boxes_ptr++, 0);
        EXPECT_EQ(*classes_ptr++, 0);
        EXPECT_EQ(*difficult_ptr++, 0);
    }
}
