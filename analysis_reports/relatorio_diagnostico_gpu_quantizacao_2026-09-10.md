# Diagnóstico da interrupção da quantização — 10/09/2026

## Conclusão

O treinamento não foi retomado porque a sessão gráfica do Mac está bloqueada. O Apple M4 e o Metal continuam presentes no sistema, e o ambiente Python contém TensorFlow 2.18.1 e `tensorflow-metal` 1.2.0. Porém, no contexto atual do terminal não interativo, o TensorFlow não recebe nenhum dispositivo GPU enquanto a sessão gráfica está bloqueada.

A correção necessária é **desbloquear o Mac na sessão gráfica do usuário** e então repetir o comando de retomada. Não há evidência de que seja necessário reinstalar TensorFlow, trocar a versão do plugin ou alterar o código do pipeline.

## Evidências coletadas

### Hardware e sistema

- macOS 15.5, arquitetura `arm64`;
- Apple iMac com chip Apple M4;
- GPU Apple M4 com 10 núcleos;
- Metal 3 suportado;
- `ioreg` enxerga o acelerador `AGXMetalG16G_B0`.

### Ambiente Python

- Python 3.10.21;
- TensorFlow 2.18.1 instalado;
- `tensorflow-metal` 1.2.0 instalado;
- `libmetal_plugin.dylib` presente e compatível com `arm64`;
- o plugin Metal chegou a ser carregado pelo `dyld` durante o diagnóstico.

### Falha reproduzida

O teste direto no ambiente atual retornou:

```text
DEVICES [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]
GPUS []
No supported GPU was found.
```

Consequentemente, a tentativa de retomada com `--resume` parou no preflight, antes de iniciar uma nova época:

```text
Preflight falhou: Nenhum dispositivo GPU TensorFlow foi detectado.
```

### Evidência da sessão bloqueada

A automação de aplicativos nativos retornou que o Mac está bloqueado e não conseguiu desbloqueá-lo automaticamente. Isso explica a diferença entre:

- o preflight anterior, que detectava `/physical_device:GPU:0` e registrava `Apple M4`;
- o preflight atual, que só detecta CPU apesar de o Metal continuar visível pelo `ioreg`.

## O que foi tentado

1. Retomada da quantização com `--resume`.
2. Preservação do escopo `--quantization-only`.
3. Preservação do batch 256 e de `--skip-litert`.
4. Verificação dos pacotes instalados e das bibliotecas Metal.
5. Teste direto de `tf.config.list_physical_devices('GPU')`.
6. Inspeção do sistema, do `ioreg`, da sessão gráfica e do carregamento do plugin.

A tentativa não alterou os resultados científicos: nenhuma nova época foi executada. Apenas o `pipeline-status.json` registrou a nova falha de preflight da tentativa de retomada.

## Como resolver

1. Desbloquear o Mac localmente, mantendo a sessão do usuário ativa.
2. Confirmar, no mesmo ambiente, que o TensorFlow retorna `/physical_device:GPU:0`.
3. Reexecutar a retomada da quantização:

```bash
cd /Users/matheussduda/repos/teste-modelos-tcc/teste-modelos-tcc

.venv-mac/bin/python scripts/run_controlled_pipeline.py \
  --suite configs/controlled-quantization-fast-mac-m4.yaml \
  --registry configs/datasets.yaml \
  --output-root outputs/controlled-quantization-fast-mac-m4 \
  --quantization-only \
  --quantization-batch-size 256 \
  --skip-litert \
  --resume
```

O comando deve preservar a MNIST FP32 concluída, repetir a MNIST FP16 falhada e continuar apenas a quantização. Ele não deve iniciar a varredura batch nem a fase de ativações.

## Status ao final da investigação

- MNIST FP32: concluída;
- MNIST FP16: falha anterior por `Broken pipe`, ainda precisa ser repetida;
- tentativa de retomada atual: bloqueada no preflight por ausência de GPU detectada;
- jobs novos executados na tentativa atual: zero;
- treinamento ativo: não;
- próximo bloqueio: desbloquear a sessão gráfica do Mac.

## Justificativa

Executar em CPU não é uma solução aceitável para este experimento: mudaria o uso de hardware, a duração e a comparabilidade com os resultados anteriores do Apple M4. Por isso a retomada foi interrompida antes do treino, preservando a validade metodológica da rodada.

*Relatório construído a partir de diagnóstico somente leitura, exceto pelo registro automático do novo estado de falha no `pipeline-status.json` durante a tentativa de retomada.*
