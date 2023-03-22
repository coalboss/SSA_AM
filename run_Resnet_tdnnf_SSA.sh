#!/bin/bash

# Set -e here so that we catch if any executable fails immediately
set -euo pipefail

chmod -R 755 ./

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=13
nj=96
train_set=train_worn_u400k
test_sets="eval_multiv1"
test_sets2=""
gmm=tri3
nnet3_affix=_train_worn_simu_u400k_cleaned_rvb
lm_suffix=

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
affix=1b   # affix for the TDNN directory name
tree_affix=
train_stage=-10
get_egs_stage=-10
decode_iter=

# training options
chunk_width=140,100,160
chunk_left_context=0
chunk_right_context=0
dropout_schedule='0,0@0.20,0.15@0.50,0'
label_delay=0
num_epochs=4

common_egs_dir=
xent_regularize=0.1

# training options
srand=0
remove_egs=false
reporting_email=

#decode options
extra_left_context=0
frames_per_chunk=
test_online_decoding=false  # if true, it will run the last decoding stage.


# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 11" if you have already
# run those things.
#local/nnet3/run_ivector_common.sh --stage $stage \
#                                  --train-set $train_set \
#                                  --test-sets "$test_sets" \
#                                  --gmm $gmm \
#                                  --nnet3-affix "$nnet3_affix" || exit 1;

# Problem: We have removed the "train_" prefix of our training set in
# the alignment directory names! Bad!

#gmm_dir=/work/asr/yhtu3/asr/s5/s5_train_worn_gss/exp/tri3_cleaned
#ali_dir=/work/asr/yhtu3/asr/s5/s5_train_worn_gss/exp/tri3_cleaned_ali_train_worn_gss_comb2s_cleaned_sp
#lores_train_data_dir=/work/asr/yhtu3/asr/s5/s5_train_worn_gss/data/train_worn_gss_comb2s_cleaned_sp
#lang=/work/asr/yhtu3/asr/s5/s5_train_worn_gss/data/lang_chain
tree_dir=/work/asr/lichai2/chime6/s5_track1/local/chain/Worn_GSS_BFmultiANDseletedarray/Speaker-and-space-aware/data/final_muti_346h_gss/tree_sp_ce
lat_dir=/work/asr/lichai2/chime6/s5_track1/local/chain/Worn_GSS_BFmultiANDseletedarray/Speaker-and-space-aware/data/final_muti_346h_gss/lattice_cnn_tdnn_Resbilstm_projectlayer512_v4_SpecAugment_Epoch6_addLinearComponent_CE_lr0.01_train_840
train_data_dir=/work/asr/lichai2/chime6/s5_track1/local/chain/Worn_GSS_BFmultiANDseletedarray/Speaker-and-space-aware/data/final_muti_346h_gss/train_840dim_worn_gss_64h_fordyliu_comb4s
train_ivector_dir=/work/asr/lichai2/chime6/s5_track1/local/chain/Worn_GSS_BFmultiANDseletedarray/Speaker-and-space-aware/data/final_muti_346h_gss/ivectors_train_840dim_worn_gss_64h_fordyliu_comb4s
dir=/work/asr/lichai2/chime6/s5_track1/exp/chain_train_worn_gss_BFmultiANDseletedarray_comb2s_cleaned_rvb/Input40kaldi40iflytek128iflytekFBANK/ResnetDyliu_12ftdnndim2048bn512_firstBN1024_SpecAugment_Epoch6_dialation


if [ $label_delay -gt 0 ]; then dir=${dir}_ld$label_delay; fi

#for f in $gmm_dir/final.mdl $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
#    $lores_train_data_dir/feats.scp $ali_dir/ali.1.gz; do
#  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
#done

if [ $stage -le 10 ]; then
  echo "$0: creating lang directory $lang with chain-type topology"
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d $lang ]; then
    if [ $lang/L.fst -nt data/lang/L.fst ]; then
      echo "$0: $lang already exists, not overwriting it; continuing"
    else
      echo "$0: $lang already exists and seems to be older than data/lang..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    cp -r data/lang $lang
    silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
  fi
fi

if [ $stage -le 11 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj ${nj} --cmd "$train_cmd" ${lores_train_data_dir} \
    data/lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 12 ]; then
  # Build a tree using our new topology.  We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.  The num-leaves is always somewhat less than the num-leaves from
  # the GMM baseline.
   if [ -f $tree_dir/final.mdl ]; then
     echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
     exit 1;
  fi
  steps/nnet3/chain/build_tree.sh \
    --frame-subsampling-factor 3 \
    --context-opts "--context-width=2 --central-position=1" \
    --cmd "$train_cmd" 3500 ${lores_train_data_dir} \
    $lang $ali_dir $tree_dir
fi

xent_regularize=0.1

if [ $stage -le 13 ]; then
  echo "************stage:13**************";
  echo "$0: creating neural net configs using the xconfig parser";

  if [ ! -d ./exp ]; then
   mkdir ./exp
  fi
  rm -rf $dir
  mkdir -p $dir

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)

  cnn_opts="l2-regularize=0.01"
  ivector_affine_opts="l2-regularize=0.01"
  tdnnf_first_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.0"
  tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
  att_opts="l2-regularize=0.01 dropout-proportion=0.1"
  linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.002"
  #lstm_opts="decay-time=40"
  nf1=64
  nf2=128
  nf3=256
  nb3=128
  fb128_nf1=32
  fb128_nf2=64
  fb128_nf3=128
  fb128_nf4=256
  fb128_nb4=128
  res_opts="bypass-source=batchnorm allow-zero-padding=false l2-regularize=0.01"
  cnn_opts="l2-regularize=0.01"
  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig

  input dim=100 name=ivector
  input dim=840 name=input
  slice-component name=mfcc40 input=input input-dim=840 output-dim=40 start-dim=0
  idct-layer name=kaldifb40 input=mfcc40 dim=40 cepstral-lifter=22 affine-transform-file=$dir/configs/idct.mat
  slice-component name=fea800 input=input input-dim=840 output-dim=800 start-dim=40
  no-op-component name=fea840 input=Append(kaldifb40,fea800)
  batchnorm-component name=fea840-bn
  spec-augment-layer name=spec-augment freq-max-proportion=0.5 time-zeroed-proportion=0.2 time-mask-max-frames=20
  
  slice-component name=fb40_part  input=spec-augment input-dim=840 output-dim=80 start-dim=0
  height-to-feature-maps-layer name=fb40bn_mask num-filters=2 height=40

  #slice-component name=fb40_part  input=spec-augment input-dim=840 output-dim=200 start-dim=0
  #height-to-feature-maps-layer name=fb40bn_mask num-filters=5 height=40
  slice-component name=fb128_part  input=spec-augment input-dim=840 output-dim=128 start-dim=200
  height-to-feature-maps-layer name=fb128bn_mask num-filters=1 height=128

  linear-component name=ivector-linear40 $ivector_affine_opts dim=200 input=ReplaceIndex(ivector, t, 0)
  batchnorm-component name=ivector40-bn target-rms=0.025
  combine-feature-maps-layer name=fb40_ivec input=Append(fb40bn_mask, ivector40-bn) num-filters1=2 num-filters2=5 height=40
  conv-layer name=conv1 $cnn_opts height-in=40 height-out=40 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=$nf1
  res-block name=res2 num-filters=$nf1 height=40 time-period=1 $res_opts
  res-block name=res3 num-filters=$nf1 height=40 time-period=1 $res_opts
  conv-layer name=conv4 $cnn_opts height-in=40 height-out=20 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=$nf2
  res-block name=res5 num-filters=$nf2 height=20 time-period=2 $res_opts
  res-block name=res6 num-filters=$nf2 height=20 time-period=2 $res_opts
  conv-layer name=conv7 $cnn_opts height-in=20 height-out=10 height-subsample-out=2 time-offsets=-2,0,2 height-offsets=-1,0,1 num-filters-out=$nf3
  res-block name=res8 num-filters=$nf3 num-bottleneck-filters=$nb3 height=10 time-period=4 $res_opts
  res-block name=res9 num-filters=$nf3 num-bottleneck-filters=$nb3 height=10 time-period=4 $res_opts
  res-block name=res10 num-filters=$nf3 num-bottleneck-filters=$nb3 height=10 time-period=4 $res_opts
  channel-average-layer name=channel-average dim=2560

  linear-component name=ivector-linear128 $ivector_affine_opts dim=128 input=ReplaceIndex(ivector, t, 0)
  batchnorm-component name=ivector128-bn target-rms=0.025
  combine-feature-maps-layer name=fb128_ivec input=Append(fb128bn_mask, ivector128-bn) num-filters1=1 num-filters2=1 height=128
  conv-layer name=fb128_conv1 $cnn_opts height-in=128 height-out=128 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=$fb128_nf1
  res-block name=fb128_res2 num-filters=$fb128_nf1 height=128 time-period=1 $res_opts
  res-block name=fb128_res3 num-filters=$fb128_nf1 height=128 time-period=1 $res_opts
  conv-layer name=fb128_conv4 $cnn_opts height-in=128 height-out=64 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=$fb128_nf2
  res-block name=fb128_res5 num-filters=$fb128_nf2 height=64 time-period=2 $res_opts
  res-block name=fb128_res6 num-filters=$fb128_nf2 height=64 time-period=2 $res_opts
  conv-layer name=fb128_conv7 $cnn_opts height-in=64 height-out=32 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=$fb128_nf3
  res-block name=fb128_res8 num-filters=$fb128_nf3 height=32 time-period=2 $res_opts
  res-block name=fb128_res9 num-filters=$fb128_nf3 height=32 time-period=2 $res_opts
  conv-layer name=fb128_conv10 $cnn_opts height-in=32 height-out=16 height-subsample-out=2 time-offsets=-2,0,2 height-offsets=-1,0,1 num-filters-out=$fb128_nf4
  res-block name=fb128_res11 num-filters=$fb128_nf4 num-bottleneck-filters=$fb128_nb4 height=16 time-period=4 $res_opts
  res-block name=fb128_res12 num-filters=$fb128_nf4 num-bottleneck-filters=$fb128_nb4 height=16 time-period=4 $res_opts
  res-block name=fb128_res13 num-filters=$fb128_nf4 num-bottleneck-filters=$fb128_nb4 height=16 time-period=4 $res_opts
  channel-average-layer name=fb128_channel-average dim=4096

  no-op-component name=concat_fb40_fb128 input=Append(channel-average,fb128_channel-average)
  tdnnf-layer name=tdnnf7 $tdnnf_first_opts dim=2048 bottleneck-dim=1024 time-stride=0
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=2048 bottleneck-dim=512 time-stride=3 dilation=4
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=2048 bottleneck-dim=512 time-stride=3 dilation=4
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=2048 bottleneck-dim=512 time-stride=3 dilation=4
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=2048 bottleneck-dim=512 time-stride=3 dilation=4
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=2048 bottleneck-dim=512 time-stride=3 dilation=4
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=2048 bottleneck-dim=512 time-stride=3 dilation=4
  tdnnf-layer name=tdnnf14 $tdnnf_opts dim=2048 bottleneck-dim=512 time-stride=3 dilation=4
  tdnnf-layer name=tdnnf15 $tdnnf_opts dim=2048 bottleneck-dim=512 time-stride=3 dilation=4
  tdnnf-layer name=tdnnf16 $tdnnf_opts dim=2048 bottleneck-dim=512 time-stride=3 dilation=4
  tdnnf-layer name=tdnnf17 $tdnnf_opts dim=2048 bottleneck-dim=512 time-stride=3 dilation=4
  tdnnf-layer name=tdnnf18 $tdnnf_opts dim=2048 bottleneck-dim=512 time-stride=3 dilation=4
  linear-component name=prefinal-l dim=256 $linear_opts
  
  ## adding the layers for chain branch
  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts small-dim=256 big-dim=2560
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts 

  # adding the layers for xent branch
  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts small-dim=256 big-dim=2560
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts 

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 14 ]; then
  #if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
  #  utils/create_split_dir.pl \
  #   /export/b0{5,6,7,8}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  #fi

#    --cmd "queue.pl --config /home/dpovey/queue_conly.conf" \
  echo "************stage:14**************";

  steps/nnet3/chain/train.py --stage -10 \
    --cmd "$train_cmd --mem 4G" \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.chunk-width $chunk_width \
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context $chunk_right_context \
    --egs.chunk-left-context-initial 0 \
    --egs.chunk-right-context-final 0 \
    --egs.opts "--frames-overlap-per-eg 0 --constrained false" \
    --trainer.num-chunk-per-minibatch 64,32 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs 6 \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 4\
    --trainer.optimization.initial-effective-lrate 0.00025 \
    --trainer.optimization.final-effective-lrate 0.000025 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir  || exit 1;

fi


if [ $stage -le 16 ]; then
  # First the options that are passed through to run_ivector_common.sh
  # (some of which are also used in this script directly).

  # The rest are configs specific to this script.  Most of the parameters
  # are just hardcoded at this level, in the commands below.
  echo "************stage:16**************";
  echo "$0: decode data..."
  # training options
  # training chunk-options
  #chunk_width=600
  # we don't need extra left/right context for TDNN systems.
  #chunk_left_context=160
  #chunk_right_context=0
  
 #  utils/mkgraph.sh \
 #      --self-loop-scale 1.0 data/lang${lm_suffix}/ \
 #      $tree_dir $tree_dir/graph${lm_suffix} || exit 1;

  #frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true
  #extra_left_context=$chunk_left_context;
  #frames_per_chunk=$chunk_width;
  ivectordir=exp/nnet3_train_worn_gss_comb2s_cleaned_rvb
  if [ ! -d $ivectordir ]; then
	mkdir -p $ivectordir
	cp -rf /work/asr/lichai2/chime6/s5_track1/exp/nnet3_train_worn_gss_comb2s_cleaned_rvb/extractor $ivectordir
  fi
  
  for data in $test_sets; do
    (
      local/nnet3/decode.sh \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --stage 0\
        --extra-left-context 0 \
        --extra-right-context 0\
        --extra-left-context-initial 0 \
        --extra-right-context-final 0 \
        --frames-per-chunk 150 --nj 32 \
        --ivector-dir $ivectordir \
        /work/asr/dyliu2/workspace/chime6/train/eva_data/dim840/${data} /work/asr/yhtu3/asr/s5/s5_train_worn_gss/data/lang $tree_dir/graph $dir
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

