# DaSiWa WAN 2.2 · SynthSeduction v9 · RunPod

このrepoの **mainはWAN 2.2専用**。repo名に`ltx23`は残っていますが、LTX用コード・workflowは`archive/ltx23/`に退避し、コンテナには含めません。旧イメージの`0.2.2`/`latest`タグは上書きしません。

## RunPod設定

| 項目 | 値 |
|---|---|
| Image | `ghcr.io/grawthings-beep/dasiwa-ltx23-runpod-comfyui:wan22-v9-cu128`（検証後はSHAタグ固定推奨） |
| GPU | RTX 4090 / 5090を想定。GPU名での拒否なし。実機生成ベンチマークは未実施 |
| CPU RAM | 64 GB以上を目安、余裕があれば96 GB以上。VRAMだけで選ばない |
| Container disk | 30 GB |
| Volume disk | 80 GB以上（Podローカル。**Network Volumeは不要**） |
| Volume mount | `/workspace` |
| HTTP port | `8188` |
| TCP port | 不要 |
| Container start command / entrypoint | 空欄。イメージの既定を使う |

環境変数の全項目は [runpod.env.example](runpod.env.example)。RunPod Secretsの鍵アイコンからHF/Civitaiトークンを紐付けます。**古いLTXテンプレートの環境変数は引き継がず、このファイルを使ってください。**

### ダウンロード元・アクセス権

1. [作者公式Hugging Face](https://huggingface.co/darksidewalker/DaSiWa-WAN2.2-I2V)で、HF_TOKENと同じアカウントから利用条件を確認・同意してください。これはユーザー本人が行う操作です。
2. `MODEL_SOURCE=auto`は公式HFのアクセス・サイズ・SHA256を確認し、利用可能ならXet。利用不可なら、Civitaiトークンを使って**同じv9ファイル**を取得します。
3. どちらにもアクセスできない場合、36 GBを取り始める前に明確なエラーで止まります。別checkpointやGGUFへのすり替えはありません。

`MODEL_SOURCE=hf`/`civitai`で取得元を固定することも可能。後者でもUMT5/VAEはComfy-Org公式HFから取得します。重み自体はGitやGHCRへ再配布しません。

## 使うworkflowは2本

ComfyUIのworkflow一覧 → `DaSiWa-WAN`。

- `01_DaSiWa_v9_I2V`: 普通の画像→動画。
- `02_DaSiWa_v9_Loop`: 同じ画像を開始/終了の条件に使うループ候補生成。終端1枚を落として80フレーム/16 fps = **5秒**。

画像をアップロードし、「Describe the motion」を書き、「Width / Height / Frames」でサイズを設定してRun。
初期値: **720×960 / 81フレーム / 16 fps / 4ステップ合計（High 2 + Low 2）/ CFG 1 / Euler + Simple / Shift 5**。
seedは比較しやすいよう固定。変化を出すときはHIGH側の`noise_seed`を変えます。

モデルには高速化蒸留が内蔵されています。**LightX2V・Lightningなどの高速化LoRAをさらに重ねないでください。** 追加の概念LoRAはこの最小構成には含めていません。LTX LoRAはWANと互換性がありません。

### ループの限界

入力画像を後処理で差し込む、逆再生で往復させる、クロスフェードする処理はありません。両端を画像条件で生成し、最後の重複フレームを除く方式です。
**位置が近くても速度や表情の変化まで一致するとは限らず、完全シームレスは保証しません。** 固定カメラ・周期的な小さめの動き・開始姿勢へ戻るプロンプトが適しています。

プロンプトは見た目の列挙より「何が、どちらへ、どれくらい、どう繰り返すか」を記述。例: `Locked-off camera. A steady breeze moves the fabric in continuous gentle waves. The subject remains in place. The motion completes one cycle and returns smoothly to the starting pose.`
CFG 1ではnegativeは通常評価されません。`closed mouth, unchanged expression`など必要な状態をpositiveへ。

## 起動が速くなる設計

- **公式PyTorch CUDA 12.8 runtime**（ベースの圧縮サイズ4.43 GB）をdigest固定。devel、Jupyter、Manager、LTX/RTX/GGUFなど不要なノードパックなし。
- ComfyUIと依存関係をビルド時に導入。起動時pip/git/コンパイル/ComfyUI全体コピーなし。`/opt/ComfyUI`から直接起動。
- 必須4ファイルのみ、**36,047,005,223 bytes（36.05 GB / 33.57 GiB）**。High/Low/UMT5/VAEすべてrevision・サイズ・SHA256固定。
- 大きい順に3ファイル並列、Xet高速モード。HTTP取得はaria2最大16分割、再開対応。
- 作業先と完成先を同じファイルシステムに置き、検証後にrename。14 GBファイルをキャッシュからもう一度コピーしない。
- 一度検証したファイルはサイズ・mtime・検証記録が変わらなければ再ハッシュ不要。新規Podは毎回取得する前提。
- 認証・空き容量・CUDAを確認してから大容量取得。未完了モデルをComfyUIへ公開しない。

**画面が開いた時間と、生成準備完了時間を混同しません。** 8188はまず状態画面。全モデルの検証が終わってからComfyUIへ切り替えます。
`/workspace/wan22/config/startup-metrics.json`にフェーズ別秒数、モデル別転送+検証時間を記録。イメージpull/unpack時間と初回推論ロード時間は別です。

回線が実効1 Gbpsなら重み36 GBの転送だけで理論上約4.8分、実効10 Gbpsなら約29秒。これにイメージ展開・ディスク・検証時間が加わります。**Network Volumeなしで数十GBの初回取得をゼロ秒にすることはできず、起動秒数の保証はしません。**

## 生成の速度と品質

- 作者配布のFP8 mixed重みを`weight_dtype=default`で読み、再量子化しません。GGUF/Q4/NVFP4への変更なし。
- 4ステップはこの蒸留モデルの推奨値。非蒸留モデルのステップを勝手に削ったものではありません。
- PyTorch標準SDPA。SageAttention/TeaCache/FP8-fast/追加蒸留LoRA/`--fast`/torch.compileは既定で使いません。近似計算や初回コンパイルコストを避けます。
- UMT5をCPU固定しません。ComfyUIの自動GPU管理とoffloadを使い、毎回キャッシュを捨てる処理もありません。
- 生成後の追加upscale/RIFE/音声生成/モザイクはこのrepoの既定経路には入れません。保存はMP4、H.264 CRF18・veryfast。保存前の生成解像度/フレーム数を下げません。
- VAEは通常decode。固定ComfyUIのOOM時tiled fallbackに任せます。タイル化は時間/画質特性が変わり得ます。

**LTX→WANで画質が同一になる保証はありません。** ここでの方針は指定WANモデルに対して追加の画質低下策を入れないことです。GPU未接続のCIでは実生成の品質・速度は評価できません。

## GPUエラー

GPU名の判定や`CUDA_VISIBLE_DEVICES`強制変更はしません。最小CUDA実演算を確認し、失敗時はデバイスerrno・libcuda・バージョンを診断JSONへ保存。重要な低レベル情報は通常ログにも出します。
8188の「診断ファイルを保存」で取得できます。JSONにはPod/GPU識別子を含むため私的に共有してください。トークンや署名付きURLはログへ出しません。
失敗画面は既定900秒保持。**これはPodを停止する機能ではなく、課金が自動停止するわけではありません。** ホストUVM/ドライバ障害をコンテナから修復したとは主張しません。

## 検証・再現性

```bash
python scripts/workflows.py --check
python -m unittest discover -s tests -v
docker build -t dasiwa-wan:test .
```

Docker buildは実際のComfyUIをCPUで起動して両workflowのノード・入力・接続型を照合し、4枚のテスト画像→MP4保存→3枚のdecodeまで確認します。WAN重みのロード・GPU推論は未検証であり、CI成功はその代わりではありません。

コード/テンプレート更新は依存関係レイヤーを再インストールせずにビルド可能。新しい公開タグは`wan22-v9-cu128`と`wan22-v9-cu128-sha-<full commit>`。既存のLTXタグは上書きしません。
旧LTXデータを`/workspace/ComfyUI`から削除する処理はありません。新WANは`/workspace/wan22`を利用し、編集済みの配布workflowはconfig内に退避してから更新します。

## 根拠・来歴

- [指定SeaArtアーカイブ](https://civarchive.com/seaart/models/8a74c3d7c313ee87c8476ee1b858bf39/versions/7dadb3c56a80037da8eaeb7f2cd0042b)
- [作者公式モデル・推奨sampler・利用条件](https://huggingface.co/darksidewalker/DaSiWa-WAN2.2-I2V)
- [High v9・SHA256・4step設定](https://civarchive.com/models/1981116?modelVersionId=2555640)
- [Low v9・SHA256](https://civarchive.com/models/1981116?modelVersionId=2555652)
- [HF Xetと環境変数](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)
- [固定ComfyUI WANノード](https://github.com/Comfy-Org/ComfyUI/blob/a7b1d39d342d102f305797fb5ba12dc304d9c1f5/comfy_extras/nodes_wan.py)
- GPU診断/状態画面とその回帰テストは同所有者のwan-animate-runpod `f63af08`から移植。
