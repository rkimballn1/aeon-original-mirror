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
#include <Python.h>

#include "etl_image.hpp"
#include "conversion.h"

#include <atomic>
#include <fstream>

using namespace std;
using namespace nervana;

namespace
{
    string get_debug_file_id();
    void write_image_with_settings(const string&                      filename,
                                   const cv::Mat&                     image,
                                   shared_ptr<augment::image::params> img_xform);
}


image::config::config(nlohmann::json js)
{
    if (js.is_null())
    {
        throw std::runtime_error("missing image config in json config");
    }

    for (auto& info : config_list)
    {
        info->parse(js);
    }
    verify_config("image", config_list, js);

    if (channel_major)
    {
        add_shape_type({channels, height, width}, {"channels", "height", "width"}, output_type);
    }
    else
    {
        add_shape_type({height, width, channels}, {"height", "width", "channels"}, output_type);
    }

    validate();
}

void image::config::validate()
{
    if (width <= 0)
    {
        throw std::invalid_argument("invalid width");
    }
    if (height <= 0)
    {
        throw std::invalid_argument("invalid height");
    }
}

/* Extract */
image::extractor::extractor(const image::config& cfg)
{
    if (!(cfg.channels == 1 || cfg.channels == 3))
    {
        std::stringstream ss;
        ss << "Unsupported number of channels in image: " << cfg.channels;
        throw std::runtime_error(ss.str());
    }
    else
    {
        _pixel_type = CV_MAKETYPE(CV_8U, cfg.channels);
        _color_mode = cfg.channels == 1 ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR;
    }
}

shared_ptr<image::decoded> image::extractor::extract(const void* inbuf, size_t insize) const
{
    cv::Mat output_img;

    // It is bad to cast away const, but opencv does not support a const Mat
    // The Mat is only used for imdecode on the next line so it is OK here
    cv::Mat input_img(1, insize, _pixel_type, (char*)inbuf);
    cv::imdecode(input_img, _color_mode, &output_img);

    auto rc = make_shared<image::decoded>();
    rc->add(output_img); // don't need to check return for single image
    return rc;
}

/* Transform:
    image::config will be a supplied bunch of params used by this provider.
    on each record, the transformer will use the config along with the supplied
    record to fill a transform_params structure which will have

    Spatial distortion params:
    randomly sampled crop box (based on params->center, params->horizontal_distortion, params->scale_pct, record size)
    randomly determined flip (based on params->flip)
    randomly sampled rotation angle (based on params->angle)

    Photometric distortion params:
    randomly sampled contrast, brightness, saturation, lighting values (based on params->cbs, lighting bounds)

*/
#include <cstdlib>

#define NP_NO_DEPRACATED_API NPY_1_7_API_VERSION

extern void call_finalize()
{
    PyGILState_Ensure();
    Py_Finalize();
}

struct call_initialize
{
    call_initialize()
    {
        Py_Initialize();
        std::atexit(call_finalize);
    }
};
static call_initialize x;

image::transformer::transformer(const image::config&)
{
}

image::transformer::~transformer() 
{ 
}

shared_ptr<image::decoded>
    image::transformer::transform(shared_ptr<augment::image::params> img_xform,
                                  shared_ptr<image::decoded>         img) const
{
    vector<cv::Mat> finalImageList;
    for (int i = 0; i < img->get_image_count(); i++)
    {
        finalImageList.push_back(transform_single_image(img_xform, img->get_image(i)));
    }

    auto rc = make_shared<image::decoded>();
    if (rc->add(finalImageList) == false)
    {
        rc = nullptr;
    }
    return rc;
}

template<typename T>
PyObject* convert(T& a);
/*
{
    return nullptr;   
}
*/
template<>
PyObject* convert<int>(int& a)
{
    PyObject* p = PyLong_FromLong(a);
    return p;
}

template<>
PyObject* convert<cv::Mat>(cv::Mat& img)
{
    cv::Mat to_convert = img.clone();
    NDArrayConverter cvt;
    PyObject* m = cvt.toNDArray(to_convert);

    return m;
}

template<typename H, typename... T>
void convert_arguments_aux(std::vector<PyObject*>& converted_args, H& head, T... tail)
{
    converted_args.push_back(convert(head));
    convert_arguments_aux<T...>(converted_args, tail...);
}

template<typename H>
void convert_arguments_aux(std::vector<PyObject*>& converted_args, H& head)
{
    converted_args.push_back(convert(head));
}
/*
template<>
void convert_arguments_aux<>(std::vector<PyObject*>&)
{
    return;
}
*/
template<typename... Args>
std::vector<PyObject*> convert_arguments(Args... args)
{
    std::vector<PyObject*> converted_args;
    convert_arguments_aux<Args...>(converted_args, args...);

    return converted_args;
}

struct gil_lock
{
private:
    PyGILState_STATE gstate;
    static int gil_init;

public:
    gil_lock()
    {
        if (!gil_lock::gil_init) {
            gil_lock::gil_init = 1;
            PyEval_InitThreads();
            PyEval_SaveThread();
        }
        gstate = PyGILState_Ensure();
    }

    ~gil_lock()
    {
        PyGILState_Release(gstate);
    }
};

int gil_lock::gil_init = 0;

template<typename... Args>
cv::Mat execute_plugin(std::string plugin_name, Args... args)
{
    PyObject* plugin_module_name;
    PyObject* plugin_module;
    PyObject* plugin_func;
    PyObject* ret_val;

    gil_lock l;

    plugin_module_name = PyString_FromString(plugin_name.c_str());
    plugin_module = PyImport_Import(plugin_module_name);

    if (!plugin_module)
    {
        PyErr_Print();
    }

    cv::Mat m;
    std::string plugin_func_name = "execute";
    
    if (plugin_module != NULL) {
        plugin_func = PyObject_GetAttrString(plugin_module, plugin_func_name.c_str());

        if (plugin_func != NULL)
            PyErr_Print();

        PyObject* arg_tuple = PyTuple_New(sizeof...(args));

        auto py_args = convert_arguments(args...);

        int idx = 0;

        for (auto a : py_args) {
            PyTuple_SetItem(arg_tuple, idx, a);
            idx++;
        }

        NDArrayConverter c;
        cv::Mat t = c.toMat(PyTuple_GetItem(arg_tuple, 0));
        cv::Mat z = c.toMat(py_args[0]);

        ret_val = PyObject_CallObject(plugin_func, arg_tuple);

        if (ret_val != NULL) {
            NDArrayConverter cvt;
            m = cvt.toMat(ret_val);
        } 
    } else {
        PyErr_Print();
    }

    return m;
}

cv::Mat execute_rotate_plugin(cv::Mat& img, int angle)
{
    NDArrayConverter cvt;

    PyObject* plugin_module_name;
    PyObject* plugin_module;
    PyObject* plugin_func;
    PyObject* ret_val;

    gil_lock l;

    plugin_module_name = PyString_FromString("rotate");
    plugin_module = PyImport_Import(plugin_module_name);

    if (!plugin_module)
    {
        PyErr_Print();
    }

    cv::Mat m;
    std::string plugin_func_name = "execute";
    
    if (plugin_module != NULL) {
        plugin_func = PyObject_GetAttrString(plugin_module, plugin_func_name.c_str());

        if (plugin_func == NULL)
            PyErr_Print();

        PyObject* arg_tuple = PyTuple_New(2);

        PyObject* py_img = cvt.toNDArray(img);
        PyObject* py_angle = PyLong_FromLong(angle);


        PyTuple_SetItem(arg_tuple, 0, py_img);
        PyTuple_SetItem(arg_tuple, 1, py_angle);

        ret_val = PyObject_CallObject(plugin_func, arg_tuple);


        if (ret_val != NULL) {
            Py_INCREF(ret_val);
            m = cvt.toMat(ret_val);
        } 
    } else {
        PyErr_Print();
    }

    return m;
}

void execute_flip_plugin(cv::Mat& out, const cv::Mat& img)
{
    NDArrayConverter cvt;

    PyObject* plugin_module_name;
    PyObject* plugin_module;
    PyObject* plugin_func;
    PyObject* ret_val;

    gil_lock l;

    plugin_module_name = PyString_FromString("flip");
    plugin_module = PyImport_Import(plugin_module_name);

    if (!plugin_module)
    {
        PyErr_Print();
    }

    cv::Mat m;
    std::string plugin_func_name = "execute";
    
    if (plugin_module != NULL) {
        plugin_func = PyObject_GetAttrString(plugin_module, plugin_func_name.c_str());

        if (plugin_func == NULL)
            PyErr_Print();

        PyObject* arg_tuple = PyTuple_New(1);
        PyObject* py_img = cvt.toNDArray(img);
        
        PyObject* py_re_img = reinterpret_cast<PyObject*>(img.refcount);

        PyTuple_SetItem(arg_tuple, 0, py_img);

        ret_val = PyObject_CallObject(plugin_func, arg_tuple);

        if (ret_val != NULL) {
            Py_INCREF(ret_val);
            out = cvt.toMat(ret_val);
        } 
    } else {
        PyErr_Print();
    }

//    return m;
}

/**
 * rotate
 * expand
 * crop
 * resize
 * distort
 * flip
 */
cv::Mat image::transformer::transform_single_image(shared_ptr<augment::image::params> img_xform,
                                                   cv::Mat& single_img) const
{
    // img_xform->dump(cout);
    cv::Mat rotatedImage;
    //image::rotate(single_img, rotatedImage, img_xform->angle);
    rotatedImage = execute_plugin("rotate", single_img, img_xform->angle);    
    
    cv::Mat expandedImage;
    if (img_xform->expand_ratio > 1.0)
        image::expand(
            rotatedImage, expandedImage, img_xform->expand_offset, img_xform->expand_size);
    else
        expandedImage    = rotatedImage;

    cv::Mat croppedImage = expandedImage(img_xform->cropbox);
    image::add_padding(croppedImage, img_xform->padding, img_xform->padding_crop_offset);

    cv::Mat resizedImage;
    image::resize(croppedImage, resizedImage, img_xform->output_size);
    photo.cbsjitter(resizedImage,
                    img_xform->contrast,
                    img_xform->brightness,
                    img_xform->saturation,
                    img_xform->hue);
    photo.lighting(resizedImage, img_xform->lighting, img_xform->color_noise_std);

    cv::Mat* finalImage = &resizedImage;
    cv::Mat  flippedImage;
    if (img_xform->flip)
    {
//        cv::flip(resizedImage, flippedImage, 1);
        flippedImage = execute_plugin("flip", resizedImage);

        finalImage = &flippedImage;
    }

    if (!img_xform->debug_output_directory.empty())
    {
        string id       = get_debug_file_id();
        string filename = img_xform->debug_output_directory + "/" + id;
        write_image_with_settings(filename, *finalImage, img_xform);
    }
    return *finalImage;
}

image::loader::loader(const image::config& cfg, bool fixed_aspect_ratio)
    : m_channel_major{cfg.channel_major}
    , m_fixed_aspect_ratio{fixed_aspect_ratio}
    , m_stype{cfg.get_shape_type()}
    , m_channels{cfg.channels}
{
}

void image::loader::load(const std::vector<void*>& outlist, shared_ptr<image::decoded> input) const
{
    char* outbuf = (char*)outlist[0];
    // TODO: Generalize this to also handle multi_crop case
    auto cv_type      = m_stype.get_otype().get_cv_type();
    auto element_size = m_stype.get_otype().get_size();
    auto img          = input->get_image(0);
    int  image_size   = img.channels() * img.total() * element_size;

    for (int i = 0; i < input->get_image_count(); i++)
    {
        auto            outbuf_i    = outbuf + (i * image_size);
        auto            input_image = input->get_image(i);
        vector<cv::Mat> source;
        vector<cv::Mat> target;
        vector<int>     from_to;

        if (m_fixed_aspect_ratio)
        {
            // zero out the output buffer as the image may not fill the canvas
            for (int j = 0; j < m_stype.get_byte_size(); j++)
            {
                outbuf[j] = 0;
            }

            vector<size_t> shape = m_stype.get_shape();
            // methods for image_var
            if (m_channel_major)
            {
                // Split into separate channels
                int      width           = shape[1];
                int      height          = shape[2];
                int      pix_per_channel = width * height;
                cv::Mat  b(width, height, CV_8U, outbuf);
                cv::Mat  g(width, height, CV_8U, outbuf + pix_per_channel);
                cv::Mat  r(width, height, CV_8U, outbuf + 2 * pix_per_channel);
                cv::Rect roi(0, 0, input_image.cols, input_image.rows);
                cv::Mat  b_roi       = b(roi);
                cv::Mat  g_roi       = g(roi);
                cv::Mat  r_roi       = r(roi);
                cv::Mat  channels[3] = {b_roi, g_roi, r_roi};
                cv::split(input_image, channels);
            }
            else
            {
                int     channels = shape[2];
                int     width    = shape[1];
                int     height   = shape[0];
                cv::Mat output(height, width, CV_8UC(channels), outbuf);
                cv::Mat target_roi = output(cv::Rect(0, 0, input_image.cols, input_image.rows));
                input_image.copyTo(target_roi);
            }
        }
        else
        {
            // methods for image
            source.push_back(input_image);
            if (m_channel_major)
            {
                for (int ch = 0; ch < m_channels; ch++)
                {
                    target.emplace_back(
                        img.size(), cv_type, (char*)(outbuf_i + ch * img.total() * element_size));
                    from_to.push_back(ch);
                    from_to.push_back(ch);
                }
            }
            else
            {
                target.emplace_back(
                    input_image.size(), CV_MAKETYPE(cv_type, m_channels), (char*)(outbuf_i));
                for (int ch = 0; ch < m_channels; ch++)
                {
                    from_to.push_back(ch);
                    from_to.push_back(ch);
                }
            }
            image::convert_mix_channels(source, target, from_to);
        }
    }
}

namespace
{
    string get_debug_file_id()
    {
        static std::atomic_uint index{0};
        unsigned int            number = index++;

        return std::to_string(number);
    }

    void write_image_with_settings(const string&                      filename,
                                   const cv::Mat&                     image,
                                   shared_ptr<augment::image::params> img_xform)
    {
        cv::imwrite(filename + ".png", image);
        ofstream ofs(filename + ".txt", ofstream::out);
        ofs << *img_xform;
        ofs.close();
    }
}
