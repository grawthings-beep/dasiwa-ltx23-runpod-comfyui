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

### 2026-09-10: 初版のCivitai保存名不具合を修正

初版commit `1d8dedf`では、aria2にstdinでURLを渡しながら保存名をコマンドラインで指定していたため、保存名が無視されていました。ダウンロード後に存在しない`model.part`を検証し、`size, SHA256 or safetensors validation failed`と表示する実装不具合です。モデルの破損やGPU異常を意味するとは限りません。

修正版はURLごとの保存名指定に変更し、**実際のaria2 + ローカルHTTPサーバー**で旧不具合の再現、新経路の取得→SHA256確認→正式配置をCI・コンテナ両方で試験します。ファイル未存在・サイズ不一致・ハッシュ不一致・ヘッダー異常も別々に表示します。

同じPodローカルVolumeが残っている場合、旧版が`.staging/<SHA256>/`に残した取得済みファイルを全体検証してから再利用します（`RECOVERED high/low`）。破損・未完了ファイルや別モデルは採用しません。**旧SHAタグのまま再起動しても修正は入りません。** 成功した新しいSHAタグへ変更してください。既存のWAN用環境変数は変更不要です。

### ダウンロード元・アクセス権

1. [作者公式Hugging Face](https://huggingface.co/darksidewalker/DaSiWa-WAN2.2-I2V)で、HF_TOKENと同じアカウントから利用条件を確認・同意してください。これはユーザー本人が行う操作です。
2. `MODEL_SOURCE=auto`は公式HFのアクセス・サイズ・SHA256を確認し、利用可能ならXet。利用不可なら、Civitaiトークンを使って**同じv9ファイル**を取得します。
3. どちらにもアクセスできない場合、36 GBを取り始める前に明確なエラーで止まります。別checkpointやGGUFへのすり替えはありません。

`MODEL_SOURCE=hf`/`civitai`で取得元を固定することも可能。後者でもUMT5/VAEはComfy-Org公式HFから取得します。重み自体はGitやGHCRへ再配布しません。

## 使うworkflowは3本

ComfyUIのworkflow一覧 → `DaSiWa-WAN`。

- `01_DaSiWa_v9_I2V`: 普通の画像→動画。
- `02_DaSiWa_v9_Loop`: 同じ画像を開始/終了の条件に使うループ候補生成。終端1枚を落として80フレーム/16 fps = **5秒**。
- `03_DaSiWa_v9_Loop_AutoMosaic`: 02と同じ生成設定＋完成フレームへの自動モザイク。元の01/02は変更なし。

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
- 自動モザイク用には別途 **18,846,815 bytes（約19 MB）** の検出モデルZIPを取得。SAM2などの大容量モデルは追加しません。初回展開後もZIPを検証元として保持します。
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
- 01/02に追加upscale/RIFE/音声生成/モザイクは入りません。03のみ生成後にモザイク。保存はMP4、H.264 CRF18・veryfast。保存前の生成解像度/フレーム数を下げません。
- VAEは通常decode。固定ComfyUIのOOM時tiled fallbackに任せます。タイル化は時間/画質特性が変わり得ます。

**LTX→WANで画質が同一になる保証はありません。** ここでの方針は指定WANモデルに対して追加の画質低下策を入れないことです。GPU未接続のCIでは実生成の品質・速度は評価できません。

## 自動モザイク（03専用）

同所有者のwan-animate-runpodの `WanAutoMosaicVideo` を移植。**入力画像には処理せず、VAE decode後の全フレームを検出→輪郭マスク→モザイク→MP4保存**します。追加LoRAは不要。

- 既定は `JUST` / confidence `0.30` / IoU `0.50` / block_size `0`（短辺に応じ自動、720なら14px）/ max_gap_frames `3`。
- 対象は以前と同じ `pussy,penis,testicles`。`anus`、`nipples`、断面・透視クラスは既定対象外。輪郭の少量拡張があるので、隣接領域に重なる可能性までは排除しません。
- JUSTは検出輪郭を少量広げる方式。WIDE/SAFEは余裕のある楕円も追加。画面固定のモザイク格子により、マスク移動でタイルが泳ぐのを抑えます。
- ループ終端の重複1枚を**先に除き**、80フレーム間で最大3フレームの短い検出抜けを継ぎ目も含め補間。Save MP4側の `trim_last_frame` はOFFのままにしてください（二重削除防止）。
- `device=auto` は生成モデルをoffloadして空きVRAMが2 GiB以上なら検出にGPUを利用。OOM時は当該フレームからCPUへ切り替え。検出を間引きません。次回生成のモデル再ロードと検出処理の時間は増え得ます。`cpu` はGPUを使わず、検出は遅くなる場合があります。
- 処理失敗は生成エラーとして停止。未処理MP4へのフォールバック保存なし。ただし**検出ゼロは処理エラーではなく、その部分は未処理のまま**です。検出漏れ・誤検出があるので出力全体の目視確認は必要です。正確な範囲や完全な隠蔽は保証しません。

検出器は [作者のAnime NSFW Detection v5.0](https://civitai.com/models/1313556?modelVersionId=2266294)（YOLO11s-seg）。作者の用途はアニメ画像で、実写・白黒漫画は適用範囲外。元ファイルはGit/GHCRへ再配布しません。

**既存WAN環境変数のままで有効**。`DOWNLOAD_MOSAIC_MODELS=1` が既定値です。Civitaiの取得権限がある既存の `CIVITAI_API_TOKEN`（または `CIVITAI_TOKEN`）が必要。配布ZIPのサイズ・SHA256・ZIP CRCを検証し、期待したPTファイルだけを安全に展開します。モデルは `/workspace/wan22/models/auto_mosaic/`、ComfyUIに同じパスを登録。読み込み時もハッシュ検証します。

起動時に検出モデルの実読み込み・クラス名・小さなCPU推論を検査してからWANの大容量取得へ進みます。初回は検出器の取得・検査分の時間が追加されますが、pip/git/追加依存のインストールは起動時に実行しません。ultralyticsとOpenCVはイメージに同梱。

01/02しか使わない場合は `DOWNLOAD_MOSAIC_MODELS=0` で検出器の取得・起動検査を省略可能。その場合、モデル未取得の03は実行できません。`DOWNLOAD_MODELS=0` は検出器も含めオフライン検証のみ。

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

Docker buildは実aria2によるHTTP取得・配置の回帰試験に加えて、実際のComfyUIをCPUで起動して3本のworkflowのノード・入力・接続型を照合し、4枚のテスト画像→モザイクノード（終端除去）→MP4保存→3枚のdecodeまで確認します。CIにはCivitai認証情報がないため、**YOLO11s-segのランダム重みで保存・読み込み・CPU推論を試験し、配布検出モデルの精度は試験しません**。合成マスクで輪郭外不変・全フレーム処理・疑似GPU OOM→CPU再試行・エラー時停止を別途検証。公式検出重みの実読み込み検査はPod起動時に実行します。WAN重みのロード・GPU推論・生成品質・実速度はCI未検証です。

コード/テンプレート更新は依存関係レイヤーを再インストールせずにビルド可能。新しい公開タグは`wan22-v9-cu128`と`wan22-v9-cu128-sha-<full commit>`。既存のLTXタグは上書きしません。
旧LTXデータを`/workspace/ComfyUI`から削除する処理はありません。新WANは`/workspace/wan22`を利用し、編集済みの配布workflowはconfig内に退避してから更新します。

## 根拠・来歴

- [指定SeaArtアーカイブ](https://civarchive.com/seaart/models/8a74c3d7c313ee87c8476ee1b858bf39/versions/7dadb3c56a80037da8eaeb7f2cd0042b)
- [作者公式モデル・推奨sampler・利用条件](https://huggingface.co/darksidewalker/DaSiWa-WAN2.2-I2V)
- [High v9・SHA256・4step設定](https://civarchive.com/models/1981116?modelVersionId=2555640)
- [Low v9・SHA256](https://civarchive.com/models/1981116?modelVersionId=2555652)
- [HF Xetと環境変数](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)
- [aria2のinput-fileと保存名指定](https://aria2.github.io/manual/en/html/aria2c.html#cmdoption-o)
- [固定ComfyUI WANノード](https://github.com/Comfy-Org/ComfyUI/blob/a7b1d39d342d102f305797fb5ba12dc304d9c1f5/comfy_extras/nodes_wan.py)
- GPU診断/状態画面とその回帰テストは同所有者のwan-animate-runpod `f63af08`から移植。
