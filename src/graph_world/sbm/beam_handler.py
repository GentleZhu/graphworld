# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os

import apache_beam as beam
import gin
import numpy as np

# Change the name of this...

from ..beam.generator_beam_handler import GeneratorBeamHandler
from ..beam.generator_config_sampler import GeneratorConfigSampler
from ..metrics.graph_metrics import GraphMetrics, NodeLabelMetrics
from ..sbm.sbm_simulator import GenerateStochasticBlockModelWithFeatures, MatchType
from ..sbm.utils import sbm_data_to_torchgeo_data, get_kclass_masks, MakePropMat, MakePi
from ..models.benchmarker import BenchmarkGNNParDo


class SampleSbmDoFn(GeneratorConfigSampler, beam.DoFn):

  def __init__(self, param_sampler_specs, marginal=False, normalize_features=True):
    super(SampleSbmDoFn, self).__init__(param_sampler_specs)
    self._marginal = marginal
    self._normalize_features = normalize_features
    self._AddSamplerFn('nvertex', self._SampleUniformInteger)
    self._AddSamplerFn('avg_degree', self._SampleUniformFloat)
    self._AddSamplerFn('feature_center_distance', self._SampleUniformFloat)
    self._AddSamplerFn('feature_dim', self._SampleUniformInteger)
    self._AddSamplerFn('edge_feature_dim', self._SampleUniformInteger)
    self._AddSamplerFn('edge_center_distance', self._SampleUniformFloat)
    self._AddSamplerFn('p_to_q_ratio', self._SampleUniformFloat)
    self._AddSamplerFn('num_clusters', self._SampleUniformInteger)
    self._AddSamplerFn('cluster_size_slope', self._SampleUniformFloat)
    self._AddSamplerFn('power_exponent', self._SampleUniformFloat)
    self._AddSamplerFn('ktrain', self._SampleUniformInteger)

  def process(self, sample_id, gen_times=5):
    """Sample and save SMB outputs given a configuration filepath.
    """
    # Avoid save_main_session in Pipeline args so the controller doesn't
    # have to import the same libraries as the workers which may be using
    # a custom container. The import will execute once then the sys.modeules
    # will be referenced to further calls.
    data = []
    generator_configs, marginal_params, fixed_params = [], [], []
    for i in range(gen_times):
        generator_config, marginal_param, fixed_param = self.SampleConfig(self._marginal)
        generator_config['generator_name'] = 'StochasticBlockModel'

        new_data = GenerateStochasticBlockModelWithFeatures(
          num_vertices=generator_config['nvertex'],
          num_edges=generator_config['nvertex'] * generator_config['avg_degree'],
          pi=MakePi(generator_config['num_clusters'], generator_config['cluster_size_slope']),
          prop_mat=MakePropMat(generator_config['num_clusters'], generator_config['p_to_q_ratio']),
          num_feature_groups=generator_config['num_clusters'],
          feature_group_match_type=MatchType.GROUPED,
          feature_center_distance=generator_config['feature_center_distance'],
          feature_dim=generator_config['feature_dim'],
          edge_center_distance=generator_config['edge_center_distance'],
          edge_feature_dim=generator_config['edge_feature_dim'],
          out_degs=np.random.power(generator_config['power_exponent'],
                                   generator_config['nvertex']),
          normalize_features=self._normalize_features
        )
        data.append(new_data)
        generator_configs.append(generator_config)       

    yield {'sample_id': sample_id,
           'marginal_param': None, # marginal_params,
           'fixed_params': [], # fixed_params,
           'generator_config': generator_configs,
           'data': data}


class WriteSbmDoFn(beam.DoFn):

  def __init__(self, output_path):
    self._output_path = output_path

  def process(self, element):
    sample_id = element['sample_id']
    config = element['generator_config']
    data = element['data']

    text_mime = 'text/plain'
    prefix = '{0:05}'.format(sample_id)
    config_object_name = os.path.join(self._output_path, prefix + '_config.txt')
    with beam.io.filesystems.FileSystems.create(config_object_name, text_mime) as f:
      buf = bytes(json.dumps(config), 'utf-8')
      f.write(buf)
      f.close()

    graph_object_name = os.path.join(self._output_path, prefix + '_graph.gt')
    with beam.io.filesystems.FileSystems.create(graph_object_name) as f:
      data.graph.save(f)
      f.close()

    graph_memberships_object_name = os.path.join(
      self._output_path, prefix + '_graph_memberships.txt')
    with beam.io.filesystems.FileSystems.create(graph_memberships_object_name, text_mime) as f:
      np.savetxt(f, data.graph_memberships)
      f.close()

    node_features_object_name = os.path.join(
      self._output_path, prefix + '_node_features.txt')
    with beam.io.filesystems.FileSystems.create(node_features_object_name, text_mime) as f:
      np.savetxt(f, data.node_features)
      f.close()

    feature_memberships_object_name = os.path.join(
      self._output_path, prefix + '_feature_membership.txt')
    with beam.io.filesystems.FileSystems.create(feature_memberships_object_name, text_mime) as f:
      np.savetxt(f, data.feature_memberships)
      f.close()

    edge_features_object_name = os.path.join(
      self._output_path, prefix + '_edge_features.txt')
    with beam.io.filesystems.FileSystems.create(edge_features_object_name, text_mime) as f:
      for edge_tuple, features in data.edge_features.items():
        buf = bytes('{0},{1},{2}'.format(edge_tuple[0], edge_tuple[1], features), 'utf-8')
        f.write(buf)
      f.close()


class ComputeSbmGraphMetrics(beam.DoFn):

  def process(self, element):
    out = element
    out['metrics'] = []
    d_len = len(element['data'])
    for i in range(d_len):
        tmp = GraphMetrics(element['data'][i].graph)
        tmp.update(NodeLabelMetrics(element['data'][i].graph, element['data'][i].graph_memberships, element['data'][i].node_features))
        out['metrics'].append(tmp)
    yield out


class ConvertToTorchGeoDataParDo(beam.DoFn):
  def __init__(self, output_path, ktrain=5, ktuning=5):
    self._output_path = output_path
    self._ktrain = ktrain
    self._ktuning = ktuning

  def process(self, element):
    sample_id = element['sample_id']
    sbm_data = element['data']
    self._ktrain = element['generator_config'][0]['ktrain']
    out = {
      'sample_id': sample_id,
      'metrics' : element['metrics'],
      'torch_data': None,
      'masks': None,
      'skipped': False,
      'generator_config': element['generator_config'],
      'marginal_param': element['marginal_param'],
      'fixed_params': element['fixed_params']
    }

    try:
      torch_data = [sbm_data_to_torchgeo_data(i) for i in sbm_data]
      out['torch_data'] = torch_data
      out['gt_data'] = [i.graph for i in sbm_data]

      torchgeo_stats = [{
        'nodes': i.num_nodes,
        'edges': i.num_edges,
        'average_node_degree': i.num_edges / i.num_nodes,
      } for i in torch_data]
      stats_object_name = os.path.join(self._output_path, '{0:05}_torch_stats.txt'.format(sample_id))
      with beam.io.filesystems.FileSystems.create(stats_object_name, 'text/plain') as f:
        buf = bytes(json.dumps(torchgeo_stats[0]), 'utf-8')
        f.write(buf)
        f.close()
    except:
      out['skipped'] = True
      print(f'failed to convert {sample_id}')
      logging.info(f'Failed to convert sbm_data to torchgeo for sample id {sample_id}')
      yield out
      return

    try:
      out['masks'] = []
      for data_i, config_i in zip(sbm_data, element['generator_config']):
          new_mask = get_kclass_masks(data_i, k_train=config_i['ktrain'], k_val=self._ktuning)
          out['masks'].append(new_mask)

      masks_object_name = os.path.join(self._output_path, '{0:05}_masks.txt'.format(sample_id))
      with beam.io.filesystems.FileSystems.create(masks_object_name, 'text/plain') as f:
        for mask in out['masks'][0]:
          np.savetxt(f, np.atleast_2d(mask.numpy()), fmt='%i', delimiter=' ')
        f.close()
    except:
      out['skipped'] = True
      print(f'failed masks {sample_id}')
      logging.info(f'Failed to sample masks for sample id {sample_id}')
      yield out
      return

    yield out


@gin.configurable
class SbmBeamHandler(GeneratorBeamHandler):

  @gin.configurable
  def __init__(self, param_sampler_specs, benchmarker_wrappers,
               marginal=False, num_tuning_rounds=1,
               tuning_metric='', tuning_metric_is_loss=False, ktrain=5, ktuning=5,
               normalize_features=True, save_tuning_results=False):
    self._sample_do_fn = SampleSbmDoFn(param_sampler_specs, marginal, normalize_features)
    self._benchmark_par_do = BenchmarkGNNParDo(benchmarker_wrappers, num_tuning_rounds,
                                               tuning_metric, tuning_metric_is_loss,
                                               save_tuning_results)
    self._metrics_par_do = ComputeSbmGraphMetrics()
    self._ktrain = ktrain
    self._ktuning = ktuning
    self._save_tuning_results = save_tuning_results

  def GetSampleDoFn(self):
    return self._sample_do_fn

  def GetWriteDoFn(self):
    return self._write_do_fn

  def GetConvertParDo(self):
    return self._convert_par_do

  def GetBenchmarkParDo(self):
    return self._benchmark_par_do

  def GetGraphMetricsParDo(self):
    return self._metrics_par_do

  def SetOutputPath(self, output_path):
    self._output_path = output_path
    self._write_do_fn = WriteSbmDoFn(output_path)
    self._convert_par_do = ConvertToTorchGeoDataParDo(output_path, self._ktrain,
                                                      self._ktuning)
    self._benchmark_par_do.SetOutputPath(output_path)
