"""Notebook presentation for the augmentation-0.5 Mac evidence package."""
from __future__ import annotations

import textwrap
import nbformat as nbf

SETUP = r'''
from pathlib import Path
import json
import platform
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator, FuncFormatter
from IPython.display import display, Markdown

relative = Path('analysis_reports/mac_training_complete_2026-09-30/evidence')
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / relative / 'runs.csv').exists()), None)
if ROOT is None:
    raise FileNotFoundError('Abra o notebook dentro do repositório ou junto ao pacote de evidências.')
DATA = ROOT / relative
runs = pd.read_csv(DATA / 'runs.csv')
epochs = pd.read_csv(DATA / 'epoch_history.csv')
classes = pd.read_csv(DATA / 'per_class_metrics.csv')
telemetry = pd.read_csv(DATA / 'telemetry_minute.csv')
checks = pd.read_csv(DATA / 'quality_checks.csv')
environments = json.loads((DATA / 'environments.json').read_text())
manifest = json.loads((DATA / 'manifest.json').read_text())
campaign_metadata = json.loads((DATA / 'campaign_metadata.json').read_text())
sources = pd.DataFrame(json.loads((DATA / 'sources.json').read_text()))
main = runs.loc[runs.stage.isin(['batch','activations','quantization'])].copy()
ptq = runs.loc[runs.stage.eq('ptq')].copy()
smoke = runs.loc[runs.stage.eq('smoke')].copy()
LABELS = {'mnist':'MNIST','fashion_mnist':'Fashion-MNIST','kmnist':'KMNIST',
          'emnist_balanced':'EMNIST Balanced','cifar10':'CIFAR-10',
          'cifar100_coarse':'CIFAR-100 coarse','svhn':'SVHN','gtsrb':'GTSRB','fer2013':'FER2013'}
ORDER = list(LABELS)
COLORS = {'batch-32':'#315C8C','batch-64':'#A47C13','batch-128':'#CA703E','batch-256':'#73783B',
          'relu':'#315C8C','sigmoid':'#A47C13','softmax':'#CA703E',
          'fp32':'#315C8C','fp16':'#A47C13','int8_ptq':'#CA703E'}
plt.rcParams.update({'figure.dpi':95,'savefig.dpi':110,'font.size':10,
                     'axes.titlesize':12,'axes.labelsize':10,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16,
                     'figure.facecolor':'white','axes.facecolor':'white'})
FIGURES = ROOT / 'analysis_reports/mac_training_complete_2026-09-30/figures'
FIGURES.mkdir(parents=True, exist_ok=True)
figure_count = 0
def show(fig, name):
    global figure_count
    figure_count += 1
    fig.savefig(FIGURES / (name + '.png'), bbox_inches='tight')
    plt.show()
    plt.close(fig)

def table(frame, columns=None):
    selected = frame[columns] if columns else frame
    display(selected.style.format(precision=4, na_rep='—').hide(axis='index'))

def fmt(value, digits=2, percent=False):
    return 'não disponível' if pd.isna(value) else f'{value * (100 if percent else 1):.{digits}f}' + ('%' if percent else '')

print('Ambiente que gerou o relatório (não confundir com o ambiente dos treinos):')
print(f'Python {platform.python_version()} | pandas {pd.__version__} | NumPy {np.__version__}')
print(f'{len(main)} treinos principais, {len(ptq)} PTQ, {len(smoke)} smoke tests; {len(epochs):,} épocas preservadas.')
'''

CHART_HELPERS = r'''
def compare_grid(stage, measures, title, name):
    subset = (runs if stage == 'quantization_ptq' else main)
    subset = subset.loc[subset.stage.isin(['quantization','ptq'])] if stage == 'quantization_ptq' else subset.loc[subset.stage.eq(stage)]
    conditions = [c for c in COLORS if c in set(subset.condition)]
    fig, axes = plt.subplots(len(measures), 1, figsize=(14, 3.5 * len(measures)), squeeze=False, layout='constrained')
    x = np.arange(len(ORDER)); width = .8 / len(conditions)
    for ax, (field, label, rate) in zip(axes.ravel(), measures):
        for j, condition in enumerate(conditions):
            values = subset.loc[subset.condition.eq(condition)].set_index('dataset')[field].reindex(ORDER)
            ax.bar(x + (j - (len(conditions)-1)/2)*width, values, width=width,
                   label=condition, color=COLORS[condition], edgecolor='#25323B', linewidth=.5,
                   hatch='//' if condition in ['softmax','batch-256','int8_ptq'] else None)
        ax.set_xticks(x, [LABELS[d] for d in ORDER], rotation=30, ha='right')
        ax.set_ylabel(label); ax.set_ylim(bottom=0)
        if rate: ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(ncol=len(conditions), loc='upper center', bbox_to_anchor=(.5,1.2), frameon=False)
    fig.suptitle(title, fontsize=15)
    show(fig, name)

def learning_grid(stage, name):
    subset = main.loc[main.stage.eq(stage)]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), layout='constrained', sharey=True)
    for ax, dataset in zip(axes.ravel(), ORDER):
        for _, row in subset.loc[subset.dataset.eq(dataset)].iterrows():
            h = epochs.loc[epochs.uid.eq(row.uid)].sort_values('epoch')
            ax.plot(h.epoch, h.val_macro_f1, label=row.condition, color=COLORS[row.condition],
                    linestyle='--' if row.condition in ['softmax','batch-256'] else '-', linewidth=1.5)
        ax.set_title(LABELS[dataset]); ax.set_xlabel('Época'); ax.set_ylabel('Macro-F1 de validação')
        ax.set_ylim(0,1.03); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f'Curvas completas de validação — {stage}', fontsize=15)
    show(fig, name)

def plot_run(uid):
    row = runs.set_index('uid').loc[uid]
    h = epochs.loc[epochs.uid.eq(uid)].sort_values('epoch')
    t = telemetry.loc[telemetry.uid.eq(uid)]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), layout='constrained')
    a = axes.ravel()
    if len(h):
        a[0].plot(h.epoch, h.loss, color='#315C8C', label='Treino')
        a[0].plot(h.epoch, h.val_loss, color='#CA703E', linestyle='--', label='Validação')
        a[0].set(xlabel='Época',ylabel='Entropia cruzada',title='Loss de treino e validação'); a[0].legend(frameon=False)
        a[0].ticklabel_format(axis='y',style='plain',useOffset=False)
        if row.condition == 'softmax':
            a[0].set_ylim(bottom=0)
        a[1].plot(h.epoch, h.accuracy, color='#73783B', label='Accuracy treino', linewidth=1)
        a[1].plot(h.epoch, h.val_accuracy, color='#315C8C', linestyle='--', label='Accuracy validação', linewidth=1)
        a[1].plot(h.epoch, h.val_macro_f1, color='#CA703E', label='Macro-F1 validação', linewidth=1.4)
        a[1].set(xlabel='Época',ylabel='Métrica',title='Aprendizado e generalização',ylim=(0,1.03))
        a[1].yaxis.set_major_formatter(PercentFormatter(1)); a[1].legend(fontsize=8,frameon=False)
        a[3].plot(h.epoch, h.epoch_seconds, color='#315C8C',label='Duração')
        a[3].set(xlabel='Época',ylabel='Segundos / época',title='Custo e throughput por época',ylim=(0,None))
        other = a[3].twinx(); other.plot(h.epoch,h.train_examples_per_second,color='#A47C13',linestyle='--')
        other.set_ylabel('Exemplos/s (tracejado)'); other.set_ylim(bottom=0); other.grid(False)
        if len(h) <= 10:
            for ax in [a[0],a[1],a[3]]: ax.set_xticks(h.epoch)
    else:
        for ax in [a[0],a[1],a[3]]:
            ax.text(.5,.5,'Histórico por época indisponível',ha='center',va='center',transform=ax.transAxes)
    scores = [row.test_accuracy,row.test_balanced_accuracy,row.test_macro_f1]
    a[2].bar(['Accuracy','Balanced acc.','Macro-F1'], scores, color='#315C8C',edgecolor='#25323B')
    a[2].set(title=f'Teste — {int(row.test_samples)} imagens' if pd.notna(row.test_samples) else 'Teste',ylabel='Métrica',ylim=(0,1.08))
    a[2].yaxis.set_major_formatter(PercentFormatter(1))
    for j,v in enumerate(scores):
        if pd.notna(v): a[2].text(j,v+.025,fmt(v,percent=True),ha='center',fontsize=9)
    if len(t):
        small_time = t.offset_seconds_mean.max() < 120
        scale = 1 if small_time else 3600
        time_label = 'Segundos desde a primeira amostra' if small_time else 'Horas desde a primeira amostra'
        for j,(_, session) in enumerate(t.groupby('session')):
            x = session.offset_seconds_mean / scale
            marker = 'o' if len(session)<3 else None
            for field,color,style,label in [('cpu_percent_mean','#315C8C','-','CPU sistema'),
                                           ('gpu_utilization_percent_mean','#CA703E','--','GPU')]:
                a[4].plot(x,session[field],color=color,linestyle=style,label=label if j==0 else None,linewidth=1,marker=marker,markersize=4)
            for field,color,style,label in [('process_rss_bytes_mean','#315C8C','-','RSS processo'),
                                           ('gpu_memory_used_bytes_mean','#CA703E','--','Memória GPU compartilhada')]:
                a[5].plot(x,session[field]/1024**3,color=color,linestyle=style,label=label if j==0 else None,linewidth=1,marker=marker,markersize=4)
        a[4].set(title='Hardware: médias em janelas UTC de 60 s',xlabel=time_label,ylabel='Utilização (%)',ylim=(0,105))
        a[5].set(title='Memória: médias de 60 s; séries não aditivas',xlabel=time_label,ylabel='GiB',ylim=(0,None))
        for ax in [a[4],a[5]]:
            ax.xaxis.set_major_locator(MaxNLocator(5))
            if small_time:
                ax.set_xlim(0,max(1,t.offset_seconds_mean.max()*1.1))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:.1f}'))
        a[4].legend(fontsize=8,frameon=False); a[5].legend(fontsize=8,frameon=False)
    else:
        labels = ['CPU sistema','GPU','RAM sistema']
        means=[row.cpu_mean_percent,row.gpu_mean_percent,row.ram_mean_percent]
        p95=[row.cpu_p95_percent,row.gpu_p95_percent,row.ram_p95_percent]
        x=np.arange(3)
        a[4].bar(x-.17,means,.34,color='#315C8C',label='Média')
        a[4].bar(x+.17,p95,.34,color='#A47C13',hatch='//',label='p95')
        a[4].set_xticks(x,labels); a[4].set(title='Resumo de hardware; sem série bruta',ylabel='Utilização (%)',ylim=(0,105));a[4].legend(frameon=False)
        x=np.arange(2)
        a[5].bar(x-.17,[row.process_rss_mean_gib,row.gpu_memory_mean_gib],.34,color='#315C8C',label='Média')
        a[5].bar(x+.17,[row.process_rss_max_gib,row.gpu_memory_max_gib],.34,color='#A47C13',hatch='//',label='Pico')
        a[5].set_xticks(x,['RSS processo','GPU compartilhada'])
        a[5].set(title='Memória: resumo; medidas não aditivas',ylabel='GiB',ylim=(0,None));a[5].legend(frameon=False)
    fig.suptitle(f"{LABELS[row.dataset]} · {row.condition} · {row.stage} · augmentation 0,5",fontsize=15)
    show(fig,'run_' + str(runs.index[runs.uid.eq(uid)][0]).zfill(3))

def explain_run(uid):
    r = runs.set_index('uid').loc[uid]
    h = epochs.loc[epochs.uid.eq(uid)].sort_values('epoch')
    text = f"**Resultado:** accuracy {fmt(r.test_accuracy,percent=True)}, balanced accuracy {fmt(r.test_balanced_accuracy,percent=True)}, Macro-F1 {fmt(r.test_macro_f1,percent=True)}, precisão macro {fmt(r.test_macro_precision,percent=True)}, recall macro {fmt(r.test_macro_recall,percent=True)}, AUC OVR {fmt(r.test_macro_ovr_auc,4)} e loss de teste {fmt(r.test_loss,4)}."
    duration = f'{fmt(r.training_seconds)} s' if r.stage == 'smoke' else f'{fmt(r.training_hours)} h'
    text += f" **Custo:** {duration} pela soma das épocas; {fmt(r.train_examples_per_second)} exemplos/s, ponderando pelo tempo; avaliação final {fmt(r.evaluation_seconds)} s."
    if len(h):
        text += f" **Curva:** melhor Macro-F1 de validação {fmt(r.best_val_macro_f1,percent=True)} na época {int(r.best_epoch)}; última época {fmt(r.final_val_macro_f1,percent=True)}. O resultado de teste refere-se ao checkpoint selecionado por validação, não necessariamente à última época."
        if r.stage != 'smoke' and r.test_balanced_accuracy <= 1 / r.num_classes + .005 and r.test_macro_f1 < .1:
            text += ' A balanced accuracy próxima a 1/n_classes e o F1 muito baixo são compatíveis com colapso para poucas classes; confirme a distribuição por classe na tabela de evidências. Maior uso de GPU não garante aprendizado.'
    same = main.loc[(main.stage.eq(r.stage)) & main.dataset.eq(r.dataset)]
    if r.stage in ['batch','activations','quantization']:
        best=same.loc[same.test_macro_f1.idxmax()]
        delta=(r.test_macro_f1-best.test_macro_f1)*100
        text += f" **Comparação no mesmo dataset e fase:** {best.condition} teve o maior Macro-F1 ({fmt(best.test_macro_f1,percent=True)}); esta condição ficou {delta:+.3f} p.p. em relação a ela. Diferenças são descritivas, com uma seed."
    text += f" **Hardware:** GPU média {fmt(r.gpu_mean_percent)}%, CPU do processo média {fmt(r.process_cpu_mean_percent)}%, RAM média {fmt(r.ram_mean_percent)}%; pico de RSS {fmt(r.process_rss_max_gib)} GiB e de memória GPU compartilhada {fmt(r.gpu_memory_max_gib)} GiB."
    if pd.notna(r.timing_delta_seconds) and abs(r.timing_delta_seconds)>1:
        text += f" **Reconciliação:** resumo de treino {fmt(r.training_seconds_reported/3600)} h versus soma das épocas {fmt(r.training_hours)} h; diferença {fmt(r.timing_delta_seconds/3600)} h. O resumo não representa todas as épocas preservadas."
    text += f" **Cobertura da telemetria:** {r.hardware_scope}. "
    if r.source_kind == 'exported_snapshot':
        text += 'Não há amostras brutas locais para reconstruir a evolução temporal; os gráficos de hardware mostram apenas as estatísticas exportadas.'
    display(Markdown(text))
    cs=classes.loc[classes.uid.eq(uid)].sort_values('f1')
    if len(cs):
        worst=cs.iloc[0]
        display(Markdown(f"**Classe com menor F1:** {worst.class_name}, F1 {fmt(worst.f1,percent=True)} e suporte {int(worst.support)} imagens. Métricas de todas as classes permanecem em `per_class_metrics.csv`."))
'''


def build_notebook(project, report, output, runs):
    notebook = nbf.v4.new_notebook()
    notebook.metadata.kernelspec = {'display_name':'Python 3 (relatório Mac)', 'language':'python','name':'python3'}
    notebook.metadata.language_info = {'name':'python','version':'3.10'}
    cells = []
    def clean(text):
        lines = text.strip('\n').splitlines()
        if len(lines) > 1 and lines[0] == lines[0].lstrip() and all(not line.strip() or line.startswith('    ') for line in lines[1:]):
            return lines[0] + '\n' + textwrap.dedent('\n'.join(lines[1:])).rstrip()
        return textwrap.dedent(text).strip()
    def md(text): cells.append(nbf.v4.new_markdown_cell(clean(text)))
    def code(text): cells.append(nbf.v4.new_code_cell(clean(text)))

    md('''# Relatório completo dos treinamentos no Mac — augmentation 0,5

    **Data do relatório:** 30/09/2026, fuso America/Sao_Paulo. As datas de coleta efetiva permanecem na proveniência.

    Este notebook reúne exclusivamente as campanhas Mac com `extra_fraction=0.5`: batch size, ativações e precisão numérica. São **81 treinamentos principais de 100 épocas**, **nove avaliações INT8 PTQ** e **nove smoke tests de duas épocas**, separados no apêndice. Os experimentos WSL/RTX 3050 e augmentation 2,0/0,0 ficam fora do escopo.

    O relatório inclui ambiente/hardware histórico, protocolo, métricas finais, curvas de aprendizado, custos, recursos por execução e interpretação individual. Os gráficos são gerados em Python e ficam salvos nas saídas das células; os CSVs compactos permitem reexecutar sem pesos, datasets ou TensorFlow.

    ## Resumo executivo

    O resumo numérico abaixo é calculado com os históricos completos, depois da reconciliação dos tempos. A seleção dos vencedores pelo teste serve como descrição retrospectiva; para selecionar hiperparâmetros em novos experimentos, use validação e preserve um teste independente.''')
    code(SETUP)
    code('''assert len(main)==81 and len(ptq)==9 and len(smoke)==9
    assert main.extra_fraction.eq(.5).all()
    assert main.status.eq('completed').all()
    assert not runs.uid.duplicated().any()
    assert not epochs.duplicated(['uid','epoch']).any()
    winners = main.loc[main.groupby(['stage','dataset']).test_macro_f1.idxmax()]
    batch_wins=winners.loc[winners.stage.eq('batch')].condition.value_counts()
    act_wins=winners.loc[winners.stage.eq('activations')].condition.value_counts()
    discrepancies=main.loc[main.timing_delta_seconds.abs()>1]
    summary_text=(f"**Cobertura:** {len(main)} treinos / {int(main.epochs_recorded.sum()):,} épocas. "
                  f"**Tempo completo de treino:** {main.training_hours.sum():.2f} h (soma das épocas, exclui PTQ e smoke). "
                  f"**Batch vencedor por Macro-F1 de teste:** {batch_wins.to_dict()}. "
                  f"**Ativação vencedora:** {act_wins.to_dict()}. "
                  f"**Tempos a reconciliar:** {len(discrepancies)} resumos diferem da soma das épocas em mais de 1 s. "
                  "Uma seed por condição: os resultados não estimam variabilidade entre repetições.")
    display(Markdown(summary_text))
    table(main.groupby('stage').agg(treinos=('uid','size'),epocas=('epochs_recorded','sum'),horas=('training_hours','sum')).reset_index())''')
    md('''## Contexto, método e definições

    - `extra_fraction=0.5` acrescenta exemplos aumentados equivalentes a 50% dos exemplos base de treino; não significa que todos os parâmetros de transformação valem 0,5. Teste e validação não recebem augmentation.
    - Seed 42, normalização `unit_interval`, `all_raw`, split estratificado 70/15/15 e treinamento do zero. A identidade de execução combina campanha, fase, dataset e condição: o `run_id` sozinho se repete entre campanhas e não é uma chave única.
    - **Batch:** 32/64/128/256; **ativações:** ReLU/Sigmoid/Softmax nas camadas ocultas, batch 256; **precisão:** FP32 (`float32`) versus FP16 misto (`mixed_float16`), batch 256. O restante dos manifests de batch é reconciliado adiante.
    - A CNN em `src/tcc_benchmark/model.py` tem cinco blocos de convoluções separáveis (32, 48, 64, 96, 128 filtros), GroupNormalization, ativação e max pooling, GAP/GMP concatenados, duas Dense de 256 e dropout 0,4. A saída são logits float32 e a perda é entropia cruzada esparsa `from_logits=True`. Softmax testado aqui é a ativação **oculta**, não a camada final de saída. Adam começa em 0,0003. Entradas de 64 px, GTSRB 128 px, canais nativos. Treinos completos usam 100 épocas; checkpoint escolhido por Macro-F1 de validação.
    - **Accuracy:** fração correta; **balanced accuracy:** recall médio por classe; **Macro-F1:** média dos F1 por classe, dando peso igual às classes; **AUC macro OVR:** discriminação one-versus-rest; **loss:** entropia cruzada. AUC alta pode coexistir com classificação argmax ruim. Não misturar Macro-F1 e accuracy em datasets desbalanceados. Nas tabelas numéricas, métricas de classificação usam frações entre 0 e 1; gráficos e interpretação exibem percentuais. Colunas terminadas em `_percent` já estão na escala 0–100.
    - **Tempo:** soma das durações por época, preservando o treino anterior às retomadas. "Completo" nos gráficos significa todas as épocas registradas; não inclui intervalos ociosos, trabalho descartado em épocas interrompidas ou tentativas sem duração preservada. `training_seconds_reported` é mantido para auditoria. Wall time é da tentativa registrada e não representa necessariamente toda a execução. Não somar wall time, duração de épocas e avaliação, pois medem intervalos diferentes.
    - **Throughput:** total de exemplos processados dividido pela soma das durações das épocas; inclui os exemplos de augmentation. A duração de época registrada pode incluir validação e callbacks, portanto não é um microbenchmark isolado da GPU.

    ### Premissas e limites

    Resultados comparáveis são apresentados dentro do mesmo dataset e fase. Não há múltiplas seeds, intervalos de confiança ou testes de significância. As campanhas companheiras têm diferentes datas e estados do sistema; o notebook não atribui diferenças entre campanhas exclusivamente à ativação ou precisão. PTQ é conversão/avaliação, com latência de inferência: não tem épocas nem custo de treino INT8.

    A análise está limitada às evidências do projeto. Alguns outputs locais são espelhos parciais; o snapshot exportado contém os 45 treinos de ativações/precisão, mas sem amostras temporais brutas. Valores ausentes aparecem como ausentes; não foram convertidos em zero.''')
    md('''## Ambiente e hardware usados nos treinamentos

    O ambiente abaixo vem dos arquivos coletados **durante o treino**, não de uma consulta ao Mac atual. O perfil do projeto é iMac M4; os registros verificam Apple M4, arm64, 10 CPUs lógicas, GPU de 10 núcleos e 16 GiB de memória unificada. A presença de uma única CPU/GPU lógica em TensorFlow não implica núcleos físicos individuais.

    O backend `apple-metal-ioreg` registra uso da GPU e memória compartilhada. RSS do processo e memória GPU podem se sobrepor: **não somar como RAM + VRAM**. CPU do processo pode ultrapassar 100% por usar múltiplos núcleos; CPU do sistema usa outra normalização. RAM percentual é de todo o sistema e inclui outros processos. Não há medição confiável de energia, potência ou temperatura nas evidências; pressão térmica `nominal` não equivale a uma temperatura medida.''')
    code('''environment_rows=[]
    for item in environments:
        e=item['environment']; h=e.get('hardware',{}); g=(h.get('gpus') or [{}])[0]
        environment_rows.append({'origem':'environment.json por run', 'sistema':e.get('platform',{}).get('system'),
                                 'arquitetura':e.get('platform',{}).get('machine'),'macOS':h.get('macos_version'),
                                 'CPU lógica':h.get('cpu_count_logical'),'GPU':g.get('name'),
                                 'núcleos GPU':g.get('gpu_core_count'),'RAM GiB':h.get('ram_total_bytes',np.nan)/1024**3,
                                 'Metal':h.get('metal_version'),'TensorFlow':e.get('packages',{}).get('tensorflow'),
                                 'tensorflow-metal':e.get('packages',{}).get('tensorflow_metal'),
                                 'Python':e.get('python',{}).get('version','').split(' (')[0],
                                 'NumPy':e.get('packages',{}).get('numpy'),'scikit-learn':e.get('packages',{}).get('scikit_learn')})
    environment_table=pd.DataFrame(environment_rows).drop_duplicates()
    table(environment_table.iloc[0].rename_axis('Característica').reset_index(name='Ambiente registrado'))
    display(Markdown('**Metadados da campanha de ativações (relatório histórico):**'))
    table(pd.Series(campaign_metadata['hardware']).rename_axis('Característica').reset_index(name='Ambiente da campanha'))
    display(Markdown('Os registros de batch documentam macOS 15.0 / Metal 3; o relatório histórico da campanha de ativações documenta macOS 15.5. Os CSVs companheiros não preservam macOS por run: a versão da campanha não foi atribuída artificialmente a cada treino.'))''')
    md('''### Configuração efetiva e controle das comparações

    Para batch, a assinatura de configuração remove apenas `batch_size` e compara os demais parâmetros de treino e o fingerprint do split dentro de cada dataset. Para os CSVs de ativações/quantização, apenas os parâmetros realmente preservados são comparados; não é possível verificar todos os campos dos manifests ausentes.''')
    code('''protocol_columns=['stage','condition','batch_size','dtype_policy','hidden_activation','extra_fraction','seed','learning_rate','max_epochs']
    table(main[protocol_columns].drop_duplicates().sort_values(['stage','condition']))
    raw_signatures=[]
    for _,r in main.loc[main.stage.eq('batch')].iterrows():
        m=json.loads((ROOT / r.source_ref).with_name('manifest.json').read_text()) if (ROOT / r.source_ref).with_name('manifest.json').exists() else None
        if m:
            c=m['config']; training=dict(c['training']);training.pop('batch_size',None)
            signature=json.dumps({'training':training,'seed':c['seed'],'normalization':c['normalization'],
                                  'balance_mode':c['balance_mode'],'split':m['split_fingerprint']},sort_keys=True)
            raw_signatures.append({'dataset':r.dataset,'assinatura':signature})
    if raw_signatures:
        signature_counts=pd.DataFrame(raw_signatures).groupby('dataset').assinatura.nunique().rename('assinaturas_sem_batch').reset_index()
        table(signature_counts)
        assert signature_counts.assinaturas_sem_batch.eq(1).all(), 'Comparação batch tem outros parâmetros diferentes'
    else:
        display(Markdown('Manifests brutos não estão neste pacote portátil; a verificação de assinatura foi feita na geração original.'))''')
    md('''## Dados, cobertura e auditoria

    Fontes primárias: manifests, `test_metrics.json`, histórico `checkpoints/epoch_metrics.csv` (sem somar a cópia em logs), ambiente e telemetria do batch. Fontes exportadas: `analysis_reports/activation_quantization_2026-09-30/data/`. Espelhos locais de 18 resultados de ativação foram reconciliados com o snapshot e não contados novamente.

    Nas execuções com CSV bruto, as médias/p95/picos de hardware deste relatório são recalculados sobre **amostras periódicas** preservadas, reduzindo o peso de eventos extras. Nos snapshots, as estatísticas são as exportadas pelo coletor, incluindo eventos. Ambas cobrem o ciclo de execução, não apenas kernels de treino. Isso limita a comparação direta de hardware entre campanhas.

    As séries temporais usam médias/min/máx por minuto UTC e sessão; o eixo mostra horas desde a primeira amostra, preservando intervalos entre retomadas. Não há interpolação entre sessões. O pico na tabela vem das amostras originais, não do gráfico suavizado de médias.''')
    code('''coverage=runs.groupby(['stage','source_kind','status']).agg(resultados=('uid','size'),epocas=('epochs_recorded','sum'),
        series_temporais=('telemetry_samples_raw',lambda s:s.notna().sum()),telemetrias_agregadas=('gpu_mean_percent',lambda s:s.notna().sum())).reset_index()
    table(coverage)
    table(checks.groupby(['check','passed']).size().rename('quantidade').reset_index())
    assert checks.loc[checks.check.str.startswith('classwise_'),'passed'].all()
    assert checks.loc[checks.check.isin(['epoch_count','epoch_sequence','snapshot_vs_local_summary']),'passed'].all()
    assert main.epochs_recorded.eq(100).all() and main.epochs_timed.eq(100).all()
    for field in ['test_accuracy','test_balanced_accuracy','test_macro_f1']:
        assert main[field].between(0,1).all()
    display(Markdown('Macro-F1 foi recalculado como a média dos F1 por classe; accuracy como recall ponderado pelo suporte. As duas reconciliações passaram em todas as 99 avaliações (incluindo PTQ/smoke). Os checks de duração divergentes são tratados na tabela abaixo.'))''')
    code('''time_audit=main.loc[main.timing_delta_seconds.abs()>1,['stage','dataset','condition','attempt','telemetry_sessions',
        'epochs_recorded','training_seconds_reported','training_seconds','timing_delta_seconds']].copy()
    for c in ['training_seconds_reported','training_seconds','timing_delta_seconds']: time_audit[c]=time_audit[c]/3600
    time_audit=time_audit.rename(columns={'training_seconds_reported':'resumo_h','training_seconds':'historico_h','timing_delta_seconds':'diferenca_h'})
    table(time_audit)
    display(Markdown('**Interpretação:** duração do resumo e soma das épocas divergem em 19 treinos principais. As retomadas com mais de uma sessão corroboram parte dessas diferenças. Os snapshots não permitem reconstruir as tentativas; nesses casos a causa exata permanece sem verificação. Os 100 valores de duração por execução permitem usar a mesma regra em todos os comparativos. Nenhum total de treino foi inferido de uma duração quase zero do resumo.'))''')
    code(CHART_HELPERS)
    md('''## Resultados de batch size

    Nove datasets × quatro batches, com Swish, precisão mista e 100 épocas. As métricas são do checkpoint selecionado por validação. As barras de tempo usam todas as épocas, incluindo as anteriores às retomadas.''')
    code("compare_grid('batch',[('test_macro_f1','Macro-F1 no teste',True),('test_accuracy','Accuracy no teste',True),('training_hours','Tempo completo de treino (h)',False)],'Batch size: qualidade e duração por dataset','batch_quality_time')")
    code('''batch=main.loc[main.stage.eq('batch')]
    batch_best=batch.loc[batch.groupby('dataset').test_macro_f1.idxmax()]
    batch_fast=batch.loc[batch.groupby('dataset').training_seconds.idxmin(),['dataset','condition','training_hours']].rename(columns={'condition':'batch_mais_rapido','training_hours':'horas_mais_rapido'})
    table(batch_best[['dataset','condition','test_macro_f1','test_accuracy','training_hours']].merge(batch_fast,on='dataset').sort_values('dataset'))
    for _,r in batch_best.sort_values('dataset').iterrows():
        fast=batch_fast.loc[batch_fast.dataset.eq(r.dataset)].iloc[0]
        display(Markdown(f"**{LABELS[r.dataset]}:** {r.condition} obteve Macro-F1 {fmt(r.test_macro_f1,percent=True)} em {fmt(r.training_hours)} h; {fast.batch_mais_rapido} teve a menor duração total ({fmt(fast.horas_mais_rapido)} h). É uma troca observada entre qualidade e custo, sem estimativa de significância."))''')
    code("compare_grid('batch',[('train_examples_per_second','Exemplos/s (total / duração)',False),('gpu_mean_percent','GPU média (%)',False),('process_rss_max_gib','Pico de RSS do processo (GiB)',False)],'Batch size: throughput e uso do hardware','batch_hardware')")
    code("learning_grid('batch','batch_learning')")
    md('''**Leitura das curvas:** batches menores e maiores alteram o número de atualizações por época e o custo observado. Oscilações e diferenças entre validação e teste devem ser lidas no mesmo dataset. Aumento de throughput por si só não estabelece melhora de Macro-F1. Os gráficos individuais adiante mostram loss, validação, custo e hardware de cada condição.''')
    md('''## Resultados de ativações ocultas

    ReLU, Sigmoid e Softmax, com batch 256 e precisão mista. A saída do modelo continua em logits; a comparação altera as ativações internas. Os resumos de hardware são exportados; não há série bruta por segundo disponível localmente.''')
    code("compare_grid('activations',[('test_macro_f1','Macro-F1 no teste',True),('test_balanced_accuracy','Balanced accuracy no teste',True),('training_hours','Tempo completo de treino (h)',False)],'Ativações: qualidade e custo por dataset','activation_quality_time')")
    code('''acts=main.loc[main.stage.eq('activations')]
    act_best=acts.loc[acts.groupby('dataset').test_macro_f1.idxmax()]
    table(act_best[['dataset','condition','test_accuracy','test_macro_f1','training_hours']].sort_values('dataset'))
    soft=acts.loc[acts.condition.eq('softmax')].copy()
    soft['recall_uniforme_referencia']=1/soft.num_classes
    table(soft[['dataset','num_classes','test_balanced_accuracy','recall_uniforme_referencia','test_macro_f1']])
    display(Markdown(f"**Observado:** Sigmoid vence por Macro-F1 em {(act_best.condition=='sigmoid').sum()} datasets e ReLU em {(act_best.condition=='relu').sum()}. Softmax tem Macro-F1 entre {fmt(soft.test_macro_f1.min(),percent=True)} e {fmt(soft.test_macro_f1.max(),percent=True)}. A balanced accuracy próxima de 1/n_classes é compatível com predição concentrada em poucas classes. O uso alto de GPU nestes treinos mede atividade, não sucesso de aprendizagem. A explicação causal exigiria analisar logits/gradientes ou novas intervenções; os gráficos não demonstram a causa."))''')
    code("learning_grid('activations','activation_learning')")
    code("compare_grid('activations',[('gpu_mean_percent','GPU média (%)',False),('process_cpu_mean_percent','CPU média do processo (%)',False),('gpu_memory_max_gib','Pico de memória GPU compartilhada (GiB)',False)],'Ativações: resumos de hardware exportados','activation_hardware')")
    md('''## FP32, FP16 misto e INT8 PTQ

    FP16 aqui significa `mixed_float16`: computação intermediária float16, variáveis e saída em float32 conforme a implementação. Não é um modelo com todos os parâmetros armazenados em 16 bits. FP32 e FP16 são 18 treinos do zero. INT8 PTQ é uma avaliação após conversão LiteRT, vinculada à mesma campanha de augmentation 0,5.

    O snapshot PTQ contém tamanho do modelo e métricas de inferência, mas não preserva todos os detalhes do dispositivo/delegate, warmup, calibração, threads ou modelo FP32 exportado equivalente. Portanto, comparar qualidade do PTQ com FP32 é uma descrição dos registros; latência PTQ não é speedup de treinamento nem demonstra vantagem sobre Metal.''')
    code("compare_grid('quantization_ptq',[('test_macro_f1','Macro-F1 no teste',True),('test_accuracy','Accuracy no teste',True)],'FP32, FP16 e INT8 PTQ: qualidade no mesmo dataset','quantization_quality')")
    code('''quant=main.loc[main.stage.eq('quantization')]
    paired=quant.loc[quant.condition.eq('fp16'),['dataset','test_macro_f1','training_seconds','train_examples_per_second','process_rss_max_gib']].merge(
        quant.loc[quant.condition.eq('fp32'),['dataset','test_macro_f1','training_seconds','train_examples_per_second','process_rss_max_gib']],on='dataset',suffixes=('_fp16','_fp32'))
    paired['delta_f1_fp16_pp']=(paired.test_macro_f1_fp16-paired.test_macro_f1_fp32)*100
    paired['razao_tempo_fp16_fp32']=paired.training_seconds_fp16/paired.training_seconds_fp32
    table(paired[['dataset','test_macro_f1_fp16','test_macro_f1_fp32','delta_f1_fp16_pp','razao_tempo_fp16_fp32']])
    for _,r in paired.iterrows():
        display(Markdown(f"**{LABELS[r.dataset]}:** FP16 altera o Macro-F1 em {r.delta_f1_fp16_pp:+.3f} p.p. versus FP32 e usa {r.razao_tempo_fp16_fp32:.2f}× o tempo completo registrado. Razão maior que 1 significa FP16 mais lento nesta execução; isso não implica o mesmo comportamento em outro hardware ou protocolo."))''')
    code("compare_grid('quantization',[('training_hours','Tempo completo de treino (h)',False),('train_examples_per_second','Exemplos/s (total / duração)',False),('process_rss_max_gib','Pico de RSS (GiB)',False)],'FP32 vs. FP16: tempo e memória por dataset','quantization_cost')")
    code("learning_grid('quantization','quantization_learning')")
    code('''ptq_table=ptq[['dataset','test_accuracy','test_macro_f1','ptq_model_bytes','ptq_batch_size','ptq_median_batch_latency_ms','ptq_p95_pass_seconds','ptq_throughput_examples_per_second']].copy()
    ptq_table['modelo_KiB']=ptq_table.ptq_model_bytes/1024
    table(ptq_table.drop(columns='ptq_model_bytes').sort_values('dataset'))
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    ordered=ptq.set_index('dataset').reindex(ORDER)
    axes[0].barh([LABELS[d] for d in ORDER],ordered.ptq_model_bytes/1024,color='#315C8C')
    axes[0].set(xlabel='Tamanho (KiB)',title='Modelo INT8 PTQ serializado')
    axes[1].barh([LABELS[d] for d in ORDER],ordered.ptq_median_batch_latency_ms,color='#CA703E')
    axes[1].set(xlabel='Latência mediana por batch (ms)',title='Inferência PTQ; batch 256')
    show(fig,'ptq_size_latency')
    display(Markdown('Tamanho refere-se ao arquivo serializado INT8, não ao RSS. Latência mediana é por batch; p95_pass_seconds é por passagem de avaliação, conforme o campo exportado. São denominadores diferentes. Sem referência de inferência FP32 equivalente não se calcula redução de tamanho ou speedup.'))''')
    md('''## Hardware: comparação e cobertura por treinamento

    Médias e p95 não representam consumo de energia. Não calcular uma média global de percentuais de campanhas com amostragem e escopo distintos. O ponto no gráfico representa uma execução, não uma relação causal entre utilização e qualidade.''')
    code('''fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    stage_colors={'batch':'#315C8C','activations':'#A47C13','quantization':'#CA703E'}
    for stage,g in main.groupby('stage'):
        axes[0].scatter(g.gpu_mean_percent,g.train_examples_per_second,color=stage_colors[stage],label=stage,alpha=.8,edgecolor='#25323B',s=35)
        axes[1].scatter(g.training_hours,g.test_macro_f1,color=stage_colors[stage],label=stage,alpha=.8,edgecolor='#25323B',s=35)
    axes[0].set(xlabel='GPU média (%)',ylabel='Exemplos/s (total / duração)',title='Atividade GPU e throughput por execução')
    axes[1].set(xlabel='Tempo completo de treino (h)',ylabel='Macro-F1 no teste',title='Qualidade versus tempo por execução')
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes: ax.legend(frameon=False)
    show(fig,'hardware_tradeoffs')
    hardware_cols=['dataset','condition','gpu_mean_percent','gpu_p95_percent','gpu_max_percent','cpu_mean_percent','process_cpu_mean_percent','ram_mean_percent','ram_max_percent','process_rss_max_gib','gpu_memory_max_gib','thermal_pressure']
    for stage in ['batch','activations','quantization']:
        display(Markdown('### Estatísticas completas — '+stage))
        table(main.loc[main.stage.eq(stage)].sort_values(['dataset','condition']),hardware_cols)
    display(Markdown('Há séries temporais locais para os 36 treinos de batch e os nove smoke tests. Para os 45 treinos de ativações/precisão há resumos exportados. Para PTQ não há telemetria de treinamento, pois a operação é uma conversão/avaliação.'))''')
    md('''## Catálogo individual: todos os 81 treinamentos principais

    Cada execução tem uma ficha, métricas exatas e seis painéis: loss, aprendizado, teste, custo por época, CPU/GPU e memória. Para batch, hardware é mostrado ao longo do tempo; para campanhas com snapshot, apenas as estatísticas realmente disponíveis são desenhadas. As diferenças entre fontes e retomadas aparecem na interpretação de cada ficha.''')
    for stage in ['batch','activations','quantization']:
        md(f'### Fichas — {stage}')
        subset = runs.loc[runs.stage.eq(stage)].copy()
        subset['dataset_order'] = subset.dataset.map({d:i for i,d in enumerate(['mnist','fashion_mnist','kmnist','emnist_balanced','cifar10','cifar100_coarse','svhn','gtsrb','fer2013'])})
        subset = subset.sort_values(['dataset_order','batch_size','condition'])
        for _, row in subset.iterrows():
            md(f"#### {row.dataset} · {row.condition}\n\nIdentidade: `{row.uid}`. Fonte: `{row.source_ref}`. Status: `{row.status}`; fonte `{row.source_kind}`.")
            code(f"plot_run({row.uid!r})\nexplain_run({row.uid!r})")
    md('''## Fichas das nove avaliações INT8 PTQ

    Sem gráficos de treino ou hardware inventados. As métricas por classe, tamanho e desempenho de inferência estão preservados na tabela de cada avaliação.''')
    for _, row in runs.loc[runs.stage.eq('ptq')].sort_values('dataset').iterrows():
        md(f"### {row.dataset} · INT8 PTQ\n\nFonte: `{row.source_ref}`. Operação de pós-treinamento vinculada à campanha de augmentation 0,5.")
        code(f'''r = ptq.set_index('uid').loc[{row.uid!r}]
        base = main.loc[(main.stage=='quantization') & (main.dataset==r.dataset) & (main.condition=='fp32')].iloc[0]
        display(Markdown(f"Accuracy {{fmt(r.test_accuracy,percent=True)}}, Macro-F1 {{fmt(r.test_macro_f1,percent=True)}}; diferença descritiva vs. FP32 {{(r.test_macro_f1-base.test_macro_f1)*100:+.3f}} p.p. Modelo {{r.ptq_model_bytes/1024:.2f}} KiB, latência mediana por batch {{r.ptq_median_batch_latency_ms:.2f}} ms e throughput {{r.ptq_throughput_examples_per_second:.2f}} exemplos/s. Não há medição de treino INT8 nem telemetria de treino nesta avaliação."))
        cs = classes.loc[classes.uid.eq(r.name)].sort_values('class_label')
        fig,ax=plt.subplots(figsize=(12,4),layout='constrained')
        ax.bar(np.arange(len(cs)),cs.f1,color='#315C8C',edgecolor='#25323B')
        ax.set_xticks(np.arange(len(cs)),cs.class_name,rotation=45,ha='right')
        ax.set(title=LABELS[r.dataset]+' · INT8 PTQ · F1 por classe',ylabel='F1',ylim=(0,1.05))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        show(fig,'ptq_classes_'+r.dataset)
        table(cs,['class_name','precision','recall','f1','support'])''')
    md('''## Apêndice: nove smoke tests Mac com augmentation 0,5

    Estes testes de duas épocas usam amostras pequenas e verificam o funcionamento da infraestrutura. Seus resultados não entram nas comparações de qualidade ou nas 81 execuções principais. O custo e uso de hardware estão documentados porque também foram treinos executados no Mac.''')
    code("table(smoke,['dataset','epochs_completed','test_samples','test_accuracy','test_macro_f1','training_seconds','gpu_mean_percent','process_rss_max_gib'])")
    for _, row in runs.loc[runs.stage.eq('smoke')].sort_values('dataset').iterrows():
        md(f"### Smoke · {row.dataset}\n\nFonte: `{row.source_ref}`. Apenas validação da infraestrutura.")
        code(f"plot_run({row.uid!r})\nexplain_run({row.uid!r})")
    md('''## Conclusões, limitações e reprodução

    O catálogo cobre todos os resultados preservados das campanhas Mac com augmentation 0,5, incluindo custo reconciliado e hardware por treino. Vencedores por teste descrevem estas execuções e não substituem uma seleção prospectiva por validação. Batch, ativação e precisão têm efeitos diferentes em cada dataset; não há um ranking universal entre bases.

    As principais limitações são uma seed por condição, ausência de telemetria bruta dos 45 treinos companheiros, escopos de amostragem diferentes, resumos de tempo inconsistentes em retomadas e detalhes incompletos do runtime PTQ. Não é possível concluir eficiência energética nem estimar variabilidade estatística a partir destes dados.

    **Arquivos para reproduzir:** este notebook e a pasta `analysis_reports/mac_training_complete_2026-09-30/evidence/`. O notebook não usa checkpoints ou datasets e não inicia treinamento. Para regenerar o pacote a partir das fontes locais: `python scripts/build_mac_training_complete_report.py`; isso exige os outputs originais de batch e os snapshots existentes. Para apenas reexecutar: `python -m jupyter nbconvert --execute --to notebook --inplace notebooks/mac_all_training_complete_report.ipynb`, em um ambiente com as dependências abaixo. Novas versões das bibliotecas podem alterar o desenho sem alterar os dados.

    Dependências usadas e hashes SHA-256 dos arquivos-fonte são registrados abaixo. As saídas do notebook já estão preenchidas, para leitura sem execução.''')
    code('''print(f'Figuras renderizadas: {figure_count}')
    print('Dependências: Python',platform.python_version(),'| pandas',pd.__version__,'| NumPy',np.__version__,'| Matplotlib',__import__('matplotlib').__version__)
    print('Snapshot:',manifest['scope'])
    print('Validação: 81 históricos completos; métricas reconciliadas por classe; nenhuma identidade/época duplicada.')
    sources.to_csv(FIGURES.parent / 'source_index.csv',index=False)
    table(sources.head(12))
    display(Markdown('O catálogo completo de arquivos e hashes está em `evidence/sources.json` e `source_index.csv`. Os campos de origem de cada run estão em `evidence/runs.csv`; todos os históricos e classes ficam nos CSVs associados.'))''')
    notebook.cells = cells
    nbf.validate(notebook)
    nbf.write(notebook, output)
    print(f'{output}: {len(cells)} cells; execute top-to-bottom before delivery')
