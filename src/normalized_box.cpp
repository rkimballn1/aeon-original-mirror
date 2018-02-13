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

#include <sstream>
#include "etl_boundingbox.hpp"
#include "util.hpp"
#include "log.hpp"

using std::ostream;
using nbox = nervana::normalized_box::box;
using nervana::almost_equal;

nbox& nbox::operator=(const nbox& b)
{
    if (&b != this)
    {
        m_xmin = b.m_xmin;
        m_ymin = b.m_ymin;
        m_xmax = b.m_xmax;
        m_ymax = b.m_ymax;
        try
        {
            throw_if_improperly_normalized();
        }
        catch (std::exception&)
        {
            ERR << "Error when assigning normalized nbox: " << b;
            throw;
        }
    }
    return *this;
}

bool nbox::is_properly_normalized() const
{
    auto is_value_normalized = [](float x) { return almost_equal_or_greater(x, 0.0f) && x < 1.0; };
    return is_value_normalized(m_xmin) && is_value_normalized(m_xmax) &&
           is_value_normalized(m_ymin) && is_value_normalized(m_ymax);
}

void nbox::throw_if_improperly_normalized() const
{
    if (!is_properly_normalized())
    {
        std::stringstream ss;
        ss << "bounding box '" << (*this) << "' is not properly normalized";
        throw std::invalid_argument(ss.str());
    }
}
