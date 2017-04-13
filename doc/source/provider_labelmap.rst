.. ---------------------------------------------------------------------------
.. Copyright 2017 Nervana Systems Inc.
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..      http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.
.. ---------------------------------------------------------------------------

label_map
=========

The complete table of configuration parameters is shown below:

.. csv-table::
   :header: "Parameter", "Default", "Description"
   :widths: 20, 10, 50
   :delim: |
   :escape: ~

   class_names | *Required* |
   name (string) | ~"~" | Name prepended to the output buffer name
   output_type (string)| ~"uint8_t~"| Output data type.
   max_classes (uint) | 100 | Maximum number of classes in output buffer

The buffers provisioned to the model are:

.. csv-table::
   :header: "Buffer Name", "Shape", "Description"
   :widths: 20, 10, 45
   :delim: |
   :escape: ~

   label_map | ``(N)`` | Class label for each example. Note that this buffer is not in one-hot format.
