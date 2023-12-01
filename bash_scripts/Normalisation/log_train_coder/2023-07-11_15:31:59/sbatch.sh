#!/bin/bash
#SBATCH --job-name=slurm-job-cse200093
#SBATCH --time 72:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40000
#SBATCH --partition gpuV100
#SBATCH --container-image /scratch/images/sparkhadoop.sqsh  --container-mounts=/export/home/$USER:/export/home/$USER,/export/home/share:/export/home/share,/data/scratch/$USER:/data/scratch/$USER --container-mount-home --container-writable --container-workdir=/
#SBATCH --output=log_train_coder/2023-07-11_15:31:59/slurm-%j-stdout.log
#SBATCH --error=log_train_coder/2023-07-11_15:31:59/slurm-%j-stderr.log
export PATH=/usr/hdp/current/accumulo-client/bin:/usr/hdp/current/atlas-server/bin:/usr/hdp/current/beacon-client/bin:/usr/hdp/current/hive-client/bin:/usr/hdp/current/falcon-client/bin:/usr/hdp/current/flume-server/bin:/usr/hdp/current/hadoop-client/bin:/usr/hdp/current/hbase-client/bin:/usr/hdp/current/hadoop-hdfs-client/bin:/usr/hdp/current/hadoop-mapreduce-client/bin:/usr/hdp/current/hadoop-yarn-client/bin:/usr/hdp/current/hive-client/bin:/usr/hdp/current/hive-hcatalog/bin:/usr/hdp/current/hive-server2/bin:/usr/hdp/current/kafka-broker/bin:/usr/hdp/current/mahout-client/bin:/usr/hdp/current/oozie-client/bin:/usr/hdp/current/oozie-server/bin:/usr/hdp/current/phoenix-client/bin:/usr/hdp/current/pig-client/bin:/usr/hdp/share/hst/hst-agent/python-wrap:/usr/hdp/current/slider-client/bin:/usr/hdp/current/sqoop-client/bin:/usr/hdp/current/sqoop-server/bin:/usr/hdp/current/storm-slider-client/bin:/usr/hdp/current/zookeeper-client/bin:/usr/hdp/current/zookeeper-server/bin:/export/home/opt/jupyterhub/conda/bin:/export/home/opt/jupyterhub/node/bin:/export/home/opt/bin:/export/home/cse200093/.user_conda/miniconda/envs/pierrenv/bin:/usr/hdp/current/accumulo-client/bin:/usr/hdp/current/atlas-server/bin:/usr/hdp/current/beacon-client/bin:/usr/hdp/current/hive-client/bin:/usr/hdp/current/falcon-client/bin:/usr/hdp/current/flume-server/bin:/usr/hdp/current/hadoop-client/bin:/usr/hdp/current/hbase-client/bin:/usr/hdp/current/hadoop-hdfs-client/bin:/usr/hdp/current/hadoop-mapreduce-client/bin:/usr/hdp/current/hadoop-yarn-client/bin:/usr/hdp/current/hive-client/bin:/usr/hdp/current/hive-hcatalog/bin:/usr/hdp/current/hive-server2/bin:/usr/hdp/current/kafka-broker/bin:/usr/hdp/current/mahout-client/bin:/usr/hdp/current/oozie-client/bin:/usr/hdp/current/oozie-server/bin:/usr/hdp/current/phoenix-client/bin:/usr/hdp/current/pig-client/bin:/usr/hdp/share/hst/hst-agent/python-wrap:/usr/hdp/current/slider-client/bin:/usr/hdp/current/sqoop-client/bin:/usr/hdp/current/sqoop-server/bin:/usr/hdp/current/storm-slider-client/bin:/usr/hdp/current/zookeeper-client/bin:/usr/hdp/current/zookeeper-server/bin:/export/home/opt/jupyterhub/conda/bin:/export/home/opt/jupyterhub/node/bin:/export/home/opt/bin:/usr/hdp/current/accumulo-client/bin:/usr/hdp/current/atlas-server/bin:/usr/hdp/current/beacon-client/bin:/usr/hdp/current/hive-client/bin:/usr/hdp/current/falcon-client/bin:/usr/hdp/current/flume-server/bin:/usr/hdp/current/hadoop-client/bin:/usr/hdp/current/hbase-client/bin:/usr/hdp/current/hadoop-hdfs-client/bin:/usr/hdp/current/hadoop-mapreduce-client/bin:/usr/hdp/current/hadoop-yarn-client/bin:/usr/hdp/current/hive-client/bin:/usr/hdp/current/hive-hcatalog/bin:/usr/hdp/current/hive-server2/bin:/usr/hdp/current/kafka-broker/bin:/usr/hdp/current/mahout-client/bin:/usr/hdp/current/oozie-client/bin:/usr/hdp/current/oozie-server/bin:/usr/hdp/current/phoenix-client/bin:/usr/hdp/current/pig-client/bin:/usr/hdp/share/hst/hst-agent/python-wrap:/usr/hdp/current/slider-client/bin:/usr/hdp/current/sqoop-client/bin:/usr/hdp/current/sqoop-server/bin:/usr/hdp/current/storm-slider-client/bin:/usr/hdp/current/zookeeper-client/bin:/usr/hdp/current/zookeeper-server/bin:/export/home/opt/jupyterhub/conda/bin:/export/home/opt/jupyterhub/node/bin:/export/home/opt/bin:/usr/hdp/current/accumulo-client/bin:/usr/hdp/current/atlas-server/bin:/usr/hdp/current/beacon-client/bin:/usr/hdp/current/hive-client/bin:/usr/hdp/current/falcon-client/bin:/usr/hdp/current/flume-server/bin:/usr/hdp/current/hadoop-client/bin:/usr/hdp/current/hbase-client/bin:/usr/hdp/current/hadoop-hdfs-client/bin:/usr/hdp/current/hadoop-mapreduce-client/bin:/usr/hdp/current/hadoop-yarn-client/bin:/usr/hdp/current/hive-client/bin:/usr/hdp/current/hive-hcatalog/bin:/usr/hdp/current/hive-server2/bin:/usr/hdp/current/kafka-broker/bin:/usr/hdp/current/mahout-client/bin:/usr/hdp/current/oozie-client/bin:/usr/hdp/current/oozie-server/bin:/usr/hdp/current/phoenix-client/bin:/usr/hdp/current/pig-client/bin:/usr/hdp/share/hst/hst-agent/python-wrap:/usr/hdp/current/slider-client/bin:/usr/hdp/current/sqoop-client/bin:/usr/hdp/current/sqoop-server/bin:/usr/hdp/current/storm-slider-client/bin:/usr/hdp/current/zookeeper-client/bin:/usr/hdp/current/zookeeper-server/bin:/export/home/opt/jupyterhub/conda/bin:/export/home/opt/jupyterhub/node/bin:/export/home/opt/bin:/usr/hdp/current/spark-2.4.3-client/bin:/usr/hdp/current/spark-2.4.3-client/bin:/usr/local/hadoop/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/export/home/opt/apps/texlive-20190227/2018/bin/x86_64-linux:/export/home/opt/apps/texlive-20190227/2018/bin/x86_64-linux:/export/home/opt/apps/texlive-20190227/2018/bin/x86_64-linux:/export/home/opt/apps/texlive-20190227/2018/bin/x86_64-linux:/export/home/cse200093/.local/bin:/export/home/cse200093/bin:$PATH
cd '/export/home/cse200093/Jacques_Bio/normalisation/py_files'
python /export/home/cse200093/Jacques_Bio/normalisation/py_files/train.py --umls_dir /export/home/cse200093/deep_mlg_normalization/resources/umls/2021AB/ --model_name_or_path /export/home/cse200093/Jacques_Bio/data_bio/coder_output  --output_dir /export/home/cse200093/Jacques_Bio/data_bio/coder_output --gradient_accumulation_steps 8 --train_batch_size 1024 --lang eng_fr