.. ---------------------------------------------------------------------------
.. Copyright 2017-2018 Intel Corporation
.. 
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..     http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.
.. ---------------------------------------------------------------------------

label
=====

The label text files should contain a single integer between ``(0, num_classes-1)``.

The configuration for this provider accepts a few parameters:

.. csv-table::
   :header: "Name", "Default", "Description"
   :widths: 20, 10, 50
   :delim: |
   :escape: ~

   name (string) | ~"~" | Name prepended to the output buffer name
   binary (bool) | False | Determines whether label is already converted to ASCII format
   output_type (string) | ~"uint32_t~" | label data type

Available output_type's are:
* int8_t
* uint8_t
* int16_t
* uint16_t
* int32_t
* uint32_t
* float
* double
* char

The buffers provisioned to the model are:

.. csv-table::
   :header: "Buffer Name", "Shape", "Description"
   :widths: 20, 10, 45
   :delim: |
   :escape: ~

   label | ``(N)`` | Class label for each example. Note that this buffer is not in one-hot format.
