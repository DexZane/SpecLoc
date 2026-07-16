# Third-party notices

SpecLoc-owned code is distributed under the Apache License 2.0 in `LICENSE`.
The following notices apply to portions adapted from third-party projects.

## MMDetection

The command-line training/testing utilities and configuration structure adapt
patterns from [MMDetection](https://github.com/open-mmlab/mmdetection).

> Copyright (c) OpenMMLab. All rights reserved.

MMDetection is licensed under the Apache License, Version 2.0. SpecLoc's
adapted files retain the OpenMMLab copyright notice and identify that they
were modified.

## cocoapi-aitod

The AI-TOD scale partitions and detection cap used by
`src/specloc/evaluation/aitodMetric.py` are derived from the official
[cocoapi-aitod](https://github.com/jwwangchn/cocoapi-aitod) evaluation toolkit.
Its upstream `license.txt` states:

> Copyright (c) 2014, Piotr Dollar and Tsung-Yi Lin
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> 1. Redistributions of source code must retain the above copyright notice,
>    this list of conditions and the following disclaimer.
> 2. Redistributions in binary form must reproduce the above copyright
>    notice, this list of conditions and the following disclaimer in the
>    documentation and/or other materials provided with the distribution.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
> ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
> LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
> CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
> SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
> INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
> CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
> ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
> POSSIBILITY OF SUCH DAMAGE.
>
> The views and conclusions contained in the software and documentation are
> those of the authors and should not be interpreted as representing official
> policies, either expressed or implied, of the FreeBSD Project.

## Materials not distributed by this repository

- RSOD images and annotations are not included. The upstream RSOD repository
  does not provide an explicit dataset license file, so redistribution is not
  authorized by SpecLoc.
- AI-TOD/AI-TOD-v2 images and annotations are not included. The AI-TOD dataset
  is published under CC BY-NC-SA 4.0 and must remain separately licensed.
- TorchVision pretrained weights are downloaded at runtime and are not
  redistributed. Their permitted use may depend on the training dataset and
  the upstream model terms.
