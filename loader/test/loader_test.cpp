/*
 Copyright 2016 Nervana Systems Inc.
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

#include "gtest/gtest.h"
#include "argtype.hpp"
#include "datagen.hpp"
#include "batchfile.hpp"

#include "params.hpp"
#include "etl_interface.hpp"
#include "etl_image.hpp"
#include "etl_label.hpp"
#include "etl_bbox.hpp"
#include "provider.hpp"
#include "json.hpp"

extern DataGen _datagen;

using namespace std;
using namespace nervana;

static param_ptr _ip1 = make_shared<image_params>();
static param_ptr _lblp1 = make_shared<label_params>();

TEST(etl, bbox) {
    // Create test metadata
    nlohmann::json j = nlohmann::json::object();
    cv::Rect r0 = cv::Rect( 0, 0, 10, 15 );
    cv::Rect r1 = cv::Rect( 10, 10, 12, 13 );
    cv::Rect r2 = cv::Rect( 100, 100, 120, 130 );
    j["boxes"] = {bbox::extractor::create_box( r0, 3 ),
                  bbox::extractor::create_box( r1, 4 ),
                  bbox::extractor::create_box( r2, 42)};
    // cout << std::setw(4) << j << endl;

    string buffer = j.dump();

    bbox::extractor extractor;
    auto data = extractor.extract( &buffer[0], buffer.size() );
    shared_ptr<bbox::decoded> decoded = static_pointer_cast<bbox::decoded>(data);
    vector<bbox::box> boxes = decoded->get_data();
    ASSERT_EQ(3,boxes.size());
    EXPECT_EQ(r0,boxes[0].rect);
    EXPECT_EQ(r1,boxes[1].rect);
    EXPECT_EQ(r2,boxes[2].rect);
    EXPECT_EQ(3,boxes[0].label);
    EXPECT_EQ(4,boxes[1].label);
    EXPECT_EQ(42,boxes[2].label);
}

TEST(myloader, argtype) {
    map<string,shared_ptr<interface_ArgType> > args = _ip1->get_args();
    ASSERT_EQ(11, args.size());

    {
        string argString = "-h 220 -w 23 -s1 0.8";
        EXPECT_TRUE(_ip1->parse(argString)) << "missing required arguments in '" << argString << "'";
        auto ie_p = make_shared<image_extractor>(_ip1);
        EXPECT_EQ(ie_p->get_channel_count(), 3);
    }
    {

        int eo = 20;
        int tsc = 8;
        int tsh = 5;

        ostringstream argStringStream;
        argStringStream << "--extract_offset " << eo << " ";
        argStringStream << "--transform_scale " << tsc << " ";
        argStringStream << "--transform_shift " << tsh << " ";

        string argString = argStringStream.str();

        EXPECT_TRUE(_lblp1->parse(argString)) << "missing required arguments in '" << argString << "'";


        BatchFileReader bf;
        auto dataFiles = _datagen.GetFiles();
        ASSERT_GT(dataFiles.size(),0);
        string batchFileName = dataFiles[0];
        bf.open(batchFileName);

        // Just get a single item
        auto data = bf.read();
        auto labels = bf.read();
        bf.close();

        auto lstg = make_shared<label_settings>();

        std::default_random_engine r_eng(0);
        static_pointer_cast<label_params>(_lblp1)->fill_settings(lstg, r_eng);

        cout << "Set scale: " << lstg->scale << " ";
        cout << "Set shift: " << lstg->shift << endl;

        int reference = ((int) (*labels)[0] + eo)* lstg->scale + lstg->shift;

        // Take the int and do provision with it.
        auto lble = make_shared<label_extractor>(_lblp1);
        auto lblt = make_shared<label_transformer>(_lblp1);

        {
            auto lbll = make_shared<label_loader>(_lblp1);

            int reference_target = reference;
            int loaded_target = 0;
            provider pp(lble, lblt, lbll);
            pp.provide(labels->data(), 4, (char *)(&loaded_target), 4, lstg);
            EXPECT_EQ(reference_target, loaded_target);
        }

        {
            // This is a float loader that loads into a float with an offset
            param_ptr flt_lbl_params = make_shared<label_params>();
            flt_lbl_params->parse("--load_dofloat true --load_offset 0.8");
            auto lbll = make_shared<label_loader>(flt_lbl_params);

            float reference_target = reference + 0.8;
            float loaded_target = 0.0;
            provider pp(lble, lblt, lbll);
            pp.provide(labels->data(), 4, (char *)(&loaded_target), 4, lstg);
            EXPECT_EQ(reference_target, loaded_target);
        }

    }

}