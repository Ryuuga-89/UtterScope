# 注意点
* 環境はuvで管理
* 適切にブランチを利用する
* mainがメインブランチ。機能ごとにブランチを切ってmainにマージする。
* 指示に不明点がある場合は実装に進まず、必ずユーザーへ確認を求めること
* `README.md`と`README.ja.md`の内容は同一に保つこと

# 技術スタック
* メイン言語：python
* パッケージ管理：uv
* CLI表示：Typer, Rich
* 内部データ管理：Pydantic
* 音声前処理：FFmpeg
* 発話区間検出：Silero VAD
* MLフレームワーク：MLX Whisper
* 話者識別：pyannote.audio
* 高度英語評価：LLM API
* HTML：Jinja2
* 開発品質保証：Ruff, ty, pytest