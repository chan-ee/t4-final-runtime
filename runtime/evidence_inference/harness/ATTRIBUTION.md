## Executive summary (read this first)

The file `official_reason_rail.py` is an unmodified copy of the published Agenthon 2026 Track 4 participant citation rail. Its SHA-256 is pinned below so local reason shape and byte-cap checks remain tied to the inspected official source. The reasons adapter uses only its standard-library reason functions; importing this file does not load an NLI model or invoke a judge. A clean local rail result is not an official reasoning score or proof of prediction quality. The wrapper resolves quoted premises from the submitted task's frozen corpus and never accepts model-authored quote offsets.

- Upstream repository: https://github.com/Agenthon-2026/track4-analysis-public
- Commit: `1c744e1d6725340643a533f436517d72b53ca0e1`
- Source: `baselines/guardrails_example/citation_rail.py`
- Source and vendored file SHA-256: `e5b3b515e2824e7671d5223b4ccb2175aa27e0969da0f7e6854cfd6b735021a3`
- Modifications: none; filename only differs.
- License: upstream MIT license, reproduced below.

```text
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
