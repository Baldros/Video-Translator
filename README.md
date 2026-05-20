# OmniVoice Translator

Sistema base de traducao de voz usando:

- ASR: transcreve o audio de entrada.
- MT: traduz a transcricao para o idioma alvo.
- OmniVoice: sintetiza a fala traduzida, com clonagem por audio de referencia ou voice design.

O OmniVoice e um TTS multilingue zero-shot. Ele nao faz traducao sozinho; este projeto adiciona a camada de ASR + traducao em volta dele.

## Instalar

Use Python 3.10+ em um ambiente virtual novo.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .
```

Para GPU NVIDIA, instale PyTorch conforme sua versao de CUDA antes do `pip install -e .`.

## CLI

```powershell
omnivoice-translate `
  --input samples\input.wav `
  --output out\translated.wav `
  --source-lang eng_Latn `
  --target-lang por_Latn `
  --target-language Portuguese `
  --ref-audio samples\voice.wav
```

Sem `--ref-audio`, o OmniVoice usa voz automatica ou `--voice-instruct`.

```powershell
omnivoice-translate `
  --input samples\input.wav `
  --output out\translated.wav `
  --target-lang spa_Latn `
  --target-language Spanish `
  --voice-instruct "female, low pitch"
```

## Clonagem direta de voz

Para testar apenas clonagem de voz, sem traducao, prefira o wrapper local
`omnivoice-speak`. Ele cria a pasta de saida automaticamente e aceita
`--target-lang` como atalho para inferir o idioma do OmniVoice:

```powershell
omnivoice-speak `
  --text "Ребята, это я... Ну, я в больших кавычках... Говорю по-русски." `
  --target-lang rus_Cyrl `
  --ref-audio AudioGuia.mp4 `
  --output out\RussoTeste.wav `
  --device cuda
```

O comando oficial do OmniVoice tambem funciona, mas ele nao tem `--target-lang`;
nele use apenas `--language`:

```powershell
New-Item -ItemType Directory -Force out

omnivoice-infer `
  --text "Este e um teste de clonagem da minha voz." `
  --ref_audio samples\minha_voz.wav `
  --ref_text "Texto exato falado no arquivo minha_voz.wav" `
  --language Portuguese `
  --output out\clone_teste.wav
```

Se aparecer `LibsndfileError: Error opening 'out\\arquivo.wav': System error`,
quase sempre significa que a pasta `out` nao existe. Crie a pasta antes.
O comando `omnivoice-translate` deste projeto cria a pasta de saida automaticamente.

## UI local

```powershell
omnivoice-translate-demo --ip 127.0.0.1 --port 7860
```

## Traducao de video v0

A camada de video usa o pipeline existente de voz e adiciona uma etapa final de
lip sync plugavel:

```text
video original
-> extracao de audio para ASR
-> ASR com timestamps
-> refino das bordas dos segmentos por energia do audio
-> merge de segmentos curtos
-> traducao por trecho
-> limpeza/adaptacao do texto para fala
-> TTS por trecho com OmniVoice
-> trim/fade/normalizacao dos trechos gerados
-> montagem do audio traduzido na timeline original
-> backend de lip sync(video original + audio traduzido)
-> video traduzido
```

Backends disponiveis:

- `wav2lip`: baseline rapido.
- `latentsync`: candidato de qualidade.
- `musetalk`: alternativa pratica para dublagem.

Os modelos de lip sync nao sao empacotados neste projeto. Baixe repositorios e
checkpoints separadamente, depois passe os caminhos para o comando.

Exemplo Wav2Lip:

```powershell
omnivoice-translate-video `
  --lip-sync-backend wav2lip `
  --input AudioGuia.mp4 `
  --output out\AudioGuia_pt.mp4 `
  --source-lang eng_Latn `
  --target-lang por_Latn `
  --target-language Portuguese `
  --wav2lip-repo E:\models\Wav2Lip `
  --wav2lip-checkpoint E:\models\Wav2Lip\checkpoints\wav2lip_gan.pth `
  --wav2lip-auto-box `
  --tts-duration-mode natural `
  --timeline-fit stretch `
  --metadata out\AudioGuia_pt.json
```

Se a deteccao automatica de rosto do Wav2Lip falhar em algum frame, use um
recorte automatico estavel calculado antes da etapa de lip sync:

```powershell
omnivoice-translate-video `
  --input AudioGuia.mp4 `
  --output out\AudioGuia_en.mp4 `
  --source-lang por_Latn `
  --target-lang eng_Latn `
  --target-language English `
  --wav2lip-repo E:\models\Wav2Lip `
  --wav2lip-checkpoint E:\models\Wav2Lip\checkpoints\wav2lip_gan.pth `
  --wav2lip-auto-box
```

Se o recorte automatico nao encontrar rosto, use `--wav2lip-box` como fallback.
O formato do `--wav2lip-box` e: topo, baixo, esquerda, direita.

Se `--ref-audio` nao for informado, o sistema recorta o `source_audio.wav`
extraido do video em clips por segmento e passa cada clip como referencia para o
OmniVoice. Isso evita passar o container MP4 inteiro para o TTS, reduz vazamento
de fala de outros trechos e deixa a referencia de voz mais proxima do trecho
sintetizado. Para voltar ao comportamento antigo, use `--reference-audio-mode
source`.

Por padrao a etapa de audio tambem usa `librosa` para:

- ajustar levemente bordas de segmentos do ASR para regioes com energia real;
- aparar silencio inicial/final dos trechos gerados pelo TTS;
- aplicar fade curto nas bordas;
- normalizar RMS entre trechos;
- limitar pico antes da montagem da timeline.
- refazer automaticamente trechos em `--tts-duration-mode natural` quando o TTS
  gera fala muito mais curta que a janela do video.

Essas heuristicas podem ser desativadas com `--no-boundary-refine` e
`--no-audio-conditioning`. O retry de TTS curto pode ser desativado com
`--no-tts-retry-short-segments`. Para ajustes finos, use opcoes como
`--boundary-refine-max-shift`, `--audio-trim-top-db`, `--audio-fade-ms`,
`--audio-target-rms`, `--audio-max-gain-db`, `--audio-peak-limit` e
`--tts-min-duration-ratio`.

Arquitetura detalhada: [docs/video_translation_architecture.md](docs/video_translation_architecture.md).
Backends de lip sync: [docs/lipsync_backends.md](docs/lipsync_backends.md).
Seguranca de modelos: [docs/model_security.md](docs/model_security.md).

Audite checkpoints locais antes de carrega-los:

```powershell
omnivoice-audit-models E:\models\Wav2Lip\checkpoints\wav2lip_gan.pth
```

Audite tambem o audio gerado. Para verificar se o audio dentro do MP4 ficou
parecido com o WAV traduzido antes do mux:

```powershell
omnivoice-audit-audio out\AudioGuia_pt.mp4 `
  --reference out\AudioGuia_pt_work\translated_audio.wav
```

Essa auditoria usa `librosa` para medir duracao, RMS, clipping, silencio,
descritores espectrais, similaridade atrasada/eco e, quando ha referencia
comparavel, correlacao de envelope e distancia log-mel. O audio original em
outro idioma nao deve ser tratado como referencia direta de qualidade para a
fala traduzida. Para comparar o video final contra o audio original, passe o
metadata da traducao; nesse modo a auditoria mede prosodia por segmento
(atividade de fala, pausas, contorno de energia e F0 com DTW) e tambem acusa
timestamps sobrepostos:

```powershell
omnivoice-audit-audio out\AudioGuia_pt.mp4 `
  --reference out\AudioGuia_pt_work\source_audio.wav `
  --metadata out\AudioGuia_pt.json `
  --sample-rate 16000
```

## Converter WAV para MP4/M4A

O OmniVoice deve gerar WAV primeiro. Depois converta para AAC em MP4 ou M4A:

```powershell
wav-to-mp4 out\clone_teste.wav out\clone_teste.mp4 --overwrite
```

Ou:

```powershell
wav-to-mp4 out\clone_teste.wav out\clone_teste.m4a --bitrate 192k --overwrite
```

Esse comando usa `ffmpeg`, entao ele precisa estar instalado e disponivel no PATH.
Se o comando `wav-to-mp4` nao aparecer depois de atualizar o projeto, rode novamente:

```powershell
pip install -e .
```

## Idiomas

O modelo de traducao padrao usa codigos NLLB, por exemplo:

- `eng_Latn`: ingles
- `por_Latn`: portugues
- `spa_Latn`: espanhol
- `fra_Latn`: frances
- `deu_Latn`: alemao
- `ita_Latn`: italiano
- `jpn_Jpan`: japones
- `kor_Hang`: coreano
- `zho_Hans`: chines simplificado

`--target-language` e separado porque e passado ao OmniVoice como dica de idioma para TTS.

## Observacoes de produto

- Para traducao em tempo real, esta base precisa evoluir para processamento por chunks, VAD e fila assíncrona.
- Para clonagem de voz, use apenas audio com consentimento explicito do falante.
- Para baixa latencia, considere Whisper menor/local, modelo de traducao menor e `--num-step 16`.
