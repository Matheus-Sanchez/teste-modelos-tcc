# Backup consolidado

## Raiz de restauração

Use `G:\repos\teste-modelos-tcc` como a única raiz de backup do projeto. Ela
deve conter o código-fonte, a documentação e todas as saídas de experimento.

| Origem | Destino consolidado | Conteúdo |
|---|---|---|
| `C:\source\repos\teste-modelos-tcc` | `G:\repos\teste-modelos-tcc` | Código, configurações, notebooks, documentação e saídas já presentes no projeto. |
| `E:\tcc-benchmark\outputs` | `G:\repos\teste-modelos-tcc\outputs` | Runs e resultados gravados no segundo SSD. |

## Convenção de pastas

- `outputs/`: resultados por experimento. Cada pasta de experimento mantém seu
  nome original para preservar a rastreabilidade de checkpoints, telemetria e
  relatórios.
- `output/`: entregáveis finais e versões portáteis, como PDFs.
- `artifacts/`: produtos auxiliares gerados por relatórios ou ferramentas.
- `docs/`: documentação de operação e restauração.

## Itens intencionalmente excluídos

Pastas chamadas `node_modules/` não são copiadas ao HDD. Elas são dependências
recriáveis por `npm install`, `npm ci`, `pnpm install` ou equivalente. Essa
exclusão reduz centenas de milhares de arquivos pequenos sem perder o código
ou as definições de dependência.

## Regra de sincronização

A sincronização é incremental: copia arquivos novos ou atualizados da origem
para o HDD e não remove arquivos do destino. Ao final, compare as duas origens
com a cópia consolidada e confirme que não há arquivos pendentes.
