# 注意点
* 環境はuvで管理
* 適切にブランチを利用する
* mainがメインブランチ。機能ごとにブランチを切ってmainにマージする。
* 指示に不明点がある場合は実装に進まず、必ずユーザーへ確認を求めること
* `README.md`と`README.ja.md`の内容は同一に保つこと
* 可読性に優れたコーディングを行うこと

# CLI表示原則
* ユーザーが現在何を待っているのかが常に分かるようにすること
* 長い処理は Homebrew 風の進捗表示にする（待ち中はスピナー、完了で ✔ に置き換え）
* 対話選択は ○/● のラジオ UI とし、↑↓（または j/k）で移動・Enter で決定する
* 進捗・選択 UI は `cli/console.py` / `cli/progress.py` / `cli/select.py` を再利用し、都度独自実装しない
* quiet モードで ML ログを抑えても、進捗表示は消えないこと（`CliStream`）

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
