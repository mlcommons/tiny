"""Tool to generate the final results speadsheet from the submission checker csv output.

The resulting excel files can be imported into google sheets.
"""
import argparse
import os
import sys
import re
import numpy as np
import pandas as pd


def get_args():
  """Parse commandline."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', required=True, help='results csv from checker')
  parser.add_argument('--version', default='1.2', help='mlperf version')
  parser.add_argument('--repository', default='submissions_tiny_v1.2', help='mlperf repository')
  args = parser.parse_args()
  return args


def main():
  args = get_args()

  df = pd.read_csv(args.input).fillna('')

  # rename some fields
  df.rename(
      columns={
          'Organization': 'Submitter',
          'Division': 'Category',
          'SystemType': 'Suite',
          'SystemName': 'System',
          'number_of_nodes': 'Nodes',
          'host_processor_model_name': 'Processor',
          'accelerator_model_name': 'Accelerator',
          'accelerators_per_node': 'a#',
          'notes': 'Notes',
          'framework': 'Software',
      },
      inplace=True)
  df.rename(columns={'Model': 'UsedModel'}, inplace=True)
  df.rename(columns={'MlperfModel': 'Model'}, inplace=True)

  # fix issues with raw data
  # df['host_processor_core_count'] = df['host_processor_core_count'].apply(
  #    lambda x: 2 if x == '2 (big); 4 (LITTLE)' else x)
  df['Availability'] = df['Availability'].apply(lambda x: 'available'
                                                if x == 'on-premise' else x)

  # cleanup counts
  # df['Accelerator'] = df['Accelerator'].apply(lambda x: x if x != '-' else '')
  # df['a#'] = df['a#'].apply(lambda x: int(x) if str(x).isnumeric() else x)
  # df['a#'] = df['a#'].apply(lambda x: x if x != 0 else '')
  # df['p#'] = df.apply(lambda x: int(x['host_processors_per_node']), axis=1)

  # details url
  base_url = f'https://github.com/mlcommons/{args.repository}/tree/main'
  df['Details'] = df.apply(
      lambda x: '=HYPERLINK("{}","details")'.format('/'.join(
          [base_url, x['Category'], x['Submitter'], 'results'])),
      axis=1)
  
  # code url
  df['Code'] = df.apply(
      lambda x: '=HYPERLINK("{}","details")'.format('/'.join(
          [base_url, x['Category'], x['Submitter'], 'code'])),
      axis=1)

  output = args.input[:-4]
  writer = pd.ExcelWriter(output + '.xlsx', engine='xlsxwriter')

  indices = {}
  indices['closed'] = [
      'ID',
      'Unique ID (e.g. for Audit)',
      'ColorKey',
      'Submitter',
      'Availability',
      'SystemDesc',
      'BoardName',
      "HostProcessorModelName", "HostProcessorFrequency", "HostProcessorMemory",
      "AcceleratorModelName", "AcceleratorFrequency", "AcceleratorMemory",
      "HardwareNotes",
      "InferenceFramework", "SoftwareLibraries", "SoftwareNotes",
  ]
  indices['open'] = indices['closed'].copy()
  indices['closed'].append('Details')
  indices['closed'].append('Code')
  indices['network'] = indices['closed'].copy()
  indices['open'].append('UsedModel')
  indices['open'].append('Accuracy')
  indices['open'].append('Details')
  indices['open'].append('Code')
  columns = [
      'Model',
      # 'Scenario',
      # 'Units',
  ]
  columns_order = [['Result','Power'],
                   ['vww', 'ic', 'kws', 'ad'],
                  #  ['SingleStream', 'MultiStream', 'Server', 'Offline'],
                  #  [
                  #      'inf./sec.',
                  #      'uJ/inf.',
                  #  ]
                   ]

  def MakeWorksheet(df, index, filter_dict, sheet_name):
    for key, value in filter_dict.items():
      if type(key) == tuple:
        key = list(key)
      df = df[value(df[key])]
    print(df)
    df = df.pivot_table(index=index, columns=columns, values=['Result', 'Power'])
    df = df.fillna('nan')
    # make sure values are interpreted as float
    df = df.astype(float)
    # when there is no power, we want an empty field
    if 'Power' in df:
      df['Power'] = df['Power'].replace(0, "")
    # results should be converted from inf./sec. to latency in millisec.
    if 'Result' in df:
      df['Result'] = 1000 / df['Result']
    if df.size == 0:
      return
    for i, order in enumerate(columns_order):
      df = df.reindex(columns=order, level=i)
    df.to_excel(writer, sheet_name=sheet_name)

  def Equal(x):
    return lambda y: y == x

  def NotEqual(x):
    return lambda y: y != x

  def Contain(x):
    return lambda y: y.str.find(x) != -1

  def And(x, y):
    return lambda z: x(z) & y(z)

  def Apply(f, *args):
    return lambda x: f(x, *args)

  def MakeUniqueID(x):
    key_list = ['Category', 'Submitter', 'BoardName']
    if x['Category'] == 'open':
      key_list.append('UsedModel')
    return '/'.join(x[key_list])

  df['Unique ID (e.g. for Audit)'] = df.apply(MakeUniqueID, axis=1)
  df['ColorKey'] = df.apply(
      lambda x: ''.join(x[['Availability', 'Submitter']]), axis=1)
  df.sort_values(
      by=[
          'Category', 'Availability', 'Submitter', 'Unique ID (e.g. for Audit)'
      ],
      inplace=True)
  id_dict = {
      key: 1 + value
      for (value,
           key) in enumerate(pd.unique(df['Unique ID (e.g. for Audit)']))
  }
  df['ID'] = df.apply(
      lambda x: '{}-{:04}'.format(args.version, id_dict[x['Unique ID (e.g. for Audit)']]),
      axis=1)

  for category in ['closed', 'open', 'network']:
    for suite in ['dummy']:
      MakeWorksheet(
          df, indices[category], {
              'Category':
                  Equal(category),
              # 'Suite':
              #     Contain(suite),
              # 'Units':
              #     And(
              #         And(NotEqual('Watts'), NotEqual('millijoules')),
              #         NotEqual('millijoules/Stream')),
          }, category)

  score_format = writer.book.add_format({'num_format': '#,##0.00'})
  bg_format = writer.book.add_format({'bg_color': '#efefef'})
  for ws in writer.book.worksheets():
    ws.set_column(1, 1, None, None, {'hidden': 1})
    ws.set_column(2, 2, None, None, {'hidden': 1})
    ws.set_column(len(indices['closed']), 100, None, score_format)
    ws.conditional_format(
        2 + len(columns), 0, 200, 100, {
            'type':
                'formula',
            'criteria':
                '=mod(countunique($c$' + str(len(columns) + 3) + ':$c' +
                str(len(columns) + 3) + '), 2) = 0',
            'format':
                bg_format,
        })

  writer.close()

  return 0


if __name__ == '__main__':
  sys.exit(main())
