"""Tests for GraphSchema utils (go/tf-gnn-api)."""

from absl.testing import parameterized
import google.protobuf.text_format as pbtext
import tensorflow as tf
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.graph import schema_utils as su
import tensorflow_gnn.proto.graph_schema_pb2 as schema_pb2
from tensorflow_gnn.utils import test_utils

_SCHEMA_SPEC_MATCHING_PAIRS = [
    dict(
        testcase_name='context_schema',
        schema_pbtxt="""
          context {
            features {
              key: "label"
              value: {
                dtype: DT_STRING
              }
            }
            features {
              key: "embedding"
              value: {
                dtype: DT_FLOAT
                shape: { dim { size: 128 } }
              }
            }
          }
          """,
        graph_spec=gt.GraphTensorSpec.from_piece_specs(
            context_spec=gt.ContextSpec.from_field_specs(
                features_spec={
                    'label':
                        tf.TensorSpec(shape=(1,), dtype=tf.string),
                    'embedding':
                        tf.TensorSpec(shape=(1, 128), dtype=tf.float32),
                },
                indices_dtype=tf.int64))),
    dict(
        testcase_name='nodes_schema',
        schema_pbtxt="""
              node_sets {
                key: 'node'
                value {
                  features {
                    key: "id"
                    value: {
                      dtype: DT_INT32
                    }
                  }
                  features {
                    key: "words"
                    value: {
                      dtype: DT_STRING
                      shape: { dim { size: -1 } }
                    }
                  }
                }
              }
              """,
        graph_spec=gt.GraphTensorSpec.from_piece_specs(
            node_sets_spec={
                'node':
                    gt.NodeSetSpec.from_field_specs(
                        sizes_spec=tf.TensorSpec(
                            shape=(1,), dtype=tf.int64),
                        features_spec={
                            'id':
                                tf.TensorSpec(
                                    shape=(None,), dtype=tf.int32),
                            'words':
                                tf.RaggedTensorSpec(
                                    shape=(None, None),
                                    dtype=tf.string,
                                    ragged_rank=1,
                                    row_splits_dtype=tf.int64),
                        })
            })),
    dict(
        testcase_name='edges_schema',
        schema_pbtxt="""
              node_sets { key: 'node'}
              edge_sets {
                key: 'edge'
                value {
                  source: 'node'
                  target: 'node'
                  features {
                    key: "weight"
                    value: {
                      dtype: DT_FLOAT
                    }
                  }
                }
              }
              """,
        graph_spec=gt.GraphTensorSpec.from_piece_specs(
            node_sets_spec={
                'node':
                    gt.NodeSetSpec.from_field_specs(
                        sizes_spec=tf.TensorSpec(
                            shape=(1,), dtype=tf.int32))
            },
            edge_sets_spec={
                'edge':
                    gt.EdgeSetSpec.from_field_specs(
                        features_spec={
                            'weight':
                                tf.TensorSpec(
                                    shape=(None,), dtype=tf.float32)
                        },
                        sizes_spec=tf.TensorSpec(
                            shape=(1,), dtype=tf.int32),
                        adjacency_spec=(
                            adj.AdjacencySpec.from_incident_node_sets(
                                source_node_set='node',
                                target_node_set='node',
                                index_spec=tf.TensorSpec(
                                    shape=(None,), dtype=tf.int32))))
            }))
]


class SchemaUtilsTest(tf.test.TestCase):

  def test_iter_sets(self):
    schema = test_utils.get_proto_resource('testdata/homogeneous/citrus.pbtxt',
                                           schema_pb2.GraphSchema())

    self.assertSetEqual(
        set([('nodes', 'fruits'), ('edges', 'tastelike')]),
        set((stype, sname) for stype, sname, _ in su.iter_sets(schema)))

    # pylint: disable=pointless-statement
    schema.context.features['mutate_this']
    self.assertSetEqual(
        set([('context', ''), ('nodes', 'fruits'), ('edges', 'tastelike')]),
        set((stype, sname) for stype, sname, _ in su.iter_sets(schema)))

  def test_iter_features(self):
    schema = test_utils.get_proto_resource('testdata/homogeneous/citrus.pbtxt',
                                           schema_pb2.GraphSchema())
    self.assertSetEqual(
        set([('nodes', 'fruits'), ('edges', 'tastelike')]),
        set((stype, sname) for stype, sname, _ in su.iter_sets(schema)))


class SchemaToGraphTensorSpecTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for Graph Tensor specification."""

  def testInvariants(self):
    schema_pbtxt = """
              node_sets { key: 'node'}
              edge_sets {
                key: 'edge'
                value {
                  source: 'node'
                  target: 'node'
                  features {
                    key: "weight"
                    value: {
                      dtype: DT_FLOAT
                    }
                  }
                }
              }
              """
    schema_pb = pbtext.Merge(schema_pbtxt, schema_pb2.GraphSchema())
    result_spec = su.create_graph_spec_from_schema_pb(
        schema_pb, indices_dtype=tf.int32)
    self.assertAllEqual(result_spec.shape, tf.TensorShape([]))
    self.assertAllEqual(result_spec.indices_dtype, tf.int32)
    self.assertAllEqual(result_spec.total_num_components, 1)

    self.assertAllEqual(list(result_spec.node_sets_spec.keys()), ['node'])
    self.assertIsNone(result_spec.node_sets_spec['node'].total_size)
    self.assertEmpty(result_spec.node_sets_spec['node'].features_spec)

    self.assertAllEqual(list(result_spec.edge_sets_spec.keys()), ['edge'])
    edge_set_spec = result_spec.edge_sets_spec['edge']
    self.assertIsNone(edge_set_spec.total_size)
    self.assertEqual(list(edge_set_spec.features_spec.keys()), ['weight'])

  @parameterized.named_parameters(_SCHEMA_SPEC_MATCHING_PAIRS)
  def testParametrized(self, schema_pbtxt: str, graph_spec: gt.GraphTensorSpec):
    schema_pb = pbtext.Merge(schema_pbtxt, schema_pb2.GraphSchema())
    result_spec = su.create_graph_spec_from_schema_pb(
        schema_pb, indices_dtype=graph_spec.indices_dtype)
    self.assertAllEqual(graph_spec, result_spec)


class GraphTensorSpecToSchemaTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for Graph Tensor specification."""

  @parameterized.named_parameters(_SCHEMA_SPEC_MATCHING_PAIRS)
  def testScalar(self, schema_pbtxt: str, graph_spec: gt.GraphTensorSpec):
    result_schema = su.create_schema_pb_from_graph_spec(graph_spec)
    expected_schema_pb = pbtext.Merge(schema_pbtxt, schema_pb2.GraphSchema())
    self.assertEqual(expected_schema_pb, result_schema)

  @parameterized.named_parameters([
      # pylint:disable=g-complex-comprehension
      dict(
          testcase_name=(data['testcase_name'] + f'_{index}'),
          schema_pbtxt=data['schema_pbtxt'],
          graph_spec=data['graph_spec'],
          batch_dims=batch_dims)
      for data in _SCHEMA_SPEC_MATCHING_PAIRS
      for index, batch_dims in enumerate([[2], [None], [3, 2], [3, None]])
  ])
  def testNotScalar(self, schema_pbtxt: str, graph_spec: gt.GraphTensorSpec,
                    batch_dims):
    for batch_dim in batch_dims:
      graph_spec = graph_spec._batch(batch_dim)  # pylint: disable=protected-access
    result_schema = su.create_schema_pb_from_graph_spec(graph_spec)
    expected_schema_pb = pbtext.Merge(schema_pbtxt, schema_pb2.GraphSchema())
    self.assertEqual(expected_schema_pb, result_schema)

  @parameterized.parameters(1, 2, None)
  def testVarNumComponents(self, num_comonents):
    graph_spec = gt.GraphTensorSpec.from_piece_specs(
        context_spec=gt.ContextSpec.from_field_specs(
            features_spec={
                'label':
                    tf.TensorSpec(shape=(num_comonents,), dtype=tf.string),
                'embedding':
                    tf.TensorSpec(shape=(num_comonents, 128), dtype=tf.float32)
            },
            shape=[]))
    result_schema = su.create_schema_pb_from_graph_spec(graph_spec)
    schema_pbtxt = """
          context {
            features {
              key: "label"
              value: {
                dtype: DT_STRING
              }
            }
            features {
              key: "embedding"
              value: {
                dtype: DT_FLOAT
                shape: { dim { size: 128 } }
              }
            }
          }
          """
    expected_schema_pb = pbtext.Merge(schema_pbtxt, schema_pb2.GraphSchema())
    self.assertEqual(expected_schema_pb, result_schema)


if __name__ == '__main__':
  tf.test.main()
