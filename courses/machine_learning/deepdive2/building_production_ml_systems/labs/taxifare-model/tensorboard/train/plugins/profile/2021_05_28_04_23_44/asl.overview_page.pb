�	�C�.�(@�C�.�(@!�C�.�(@	Y�k��?Y�k��?!Y�k��?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�C�.�(@��ۻ�?A�t{I3(@YC7�嶵?*	`��"[�@2�
dIterator::Model::Prefetch::Map::Prefetch::Map::BatchV2::ShuffleAndRepeat::LegacyParallelInterleaveV2�,�"���?!ٵΗ!�J@)�,�"���?1ٵΗ!�J@:Preprocessing2
HIterator::Model::Prefetch::Map::Prefetch::Map::BatchV2::ShuffleAndRepeat��9��?!��6�V@)� ��z�?1M�u�JRB@:Preprocessing2�
lIterator::Model::Prefetch::Map::Prefetch::Map::BatchV2::ShuffleAndRepeat::LegacyParallelInterleaveV2[0]::CSVwKr��&�?!�� �@)wKr��&�?1�� �@:Preprocessing2m
6Iterator::Model::Prefetch::Map::Prefetch::Map::BatchV2�f����?!"
k)�"W@) �)U��?1�q�A@:Preprocessing2F
Iterator::Model���J��?!镕g�?)+0du��?1���{*�?:Preprocessing2U
Iterator::Model::Prefetch::Map(`;�O�?!g;�i+Y�?)�o�DIH�?1{�(����?:Preprocessing2P
Iterator::Model::PrefetchR~R���?!�%p	%��?)R~R���?1�%p	%��?:Preprocessing2_
(Iterator::Model::Prefetch::Map::Prefetch��1��?!٩by_r�?)��1��?1٩by_r�?:Preprocessing2d
-Iterator::Model::Prefetch::Map::Prefetch::Map�/����?!»�BW@)��[;Q�?1L��HVh�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9Y�k��?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��ۻ�?��ۻ�?!��ۻ�?      ��!       "      ��!       *      ��!       2	�t{I3(@�t{I3(@!�t{I3(@:      ��!       B      ��!       J	C7�嶵?C7�嶵?!C7�嶵?R      ��!       Z	C7�嶵?C7�嶵?!C7�嶵?JCPU_ONLYYY�k��?b Y      Y@qa�a�Cr�?"�
device�Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 