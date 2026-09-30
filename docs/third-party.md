# Third-party notices

Project code is copyright Team OpenVPI and uses the MIT license in `LICENSE`.

## cpp-pinyin

The pure Python engine, dictionary loaders and tone utilities are ports of
[wolfgitpr/cpp-pinyin](https://github.com/wolfgitpr/cpp-pinyin), distributed
under Apache-2.0. See `licenses/cpp-pinyin-LICENSE.txt`.

The bundled Mandarin data credit CC-CEDICT/MDBG and retain CC-BY-SA-4.0;
Cantonese data credit CC-Canto/Pleco and retain CC-BY-SA-3.0. Their existing
provenance and license notices remain beside the data in the installed package.

## cpp-kana

The Kana-to-romaji table and conversion rules in the Japanese converter were
adapted from [wolfgitpr/cpp-kana](https://github.com/wolfgitpr/cpp-kana), which
uses Apache-2.0. See `licenses/cpp-kana-LICENSE.txt`.

## External resources

UniDic is obtained through its own package/downloader or provided by the caller;
its data are not in this distribution. ONNX models, including
[LstmG2p](https://github.com/wolfgitpr/LstmG2p) models, and the caller's phoneme
mapping dictionaries are also external and retain their own licenses.
