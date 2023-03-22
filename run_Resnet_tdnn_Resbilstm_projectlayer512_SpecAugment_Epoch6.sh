#!/bin/bash
# cnn_tdnn_lstm_1c is based on cnn_tdnn_lstm_1b, but using smaller dropout-schedule
# and larger decay-time option(40).

# After testing different combinations of dropout-schedule('0,0@0.20,0.15@0.50,0',
# '0,0@0.20,0.3@0.50,0' or without), proportional-shrink(with or without) and 
# decay-time option(20, 40 or without), we found this setup is best.
# Set -e here so that we catch if any executable fails immediately
set -euo pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=13
nj=96
train_set=train_worn_u400k
test_sets="dev_gss_multiarray eval_gss_multiarray"
test_sets2="eval_gss_ali_1024_256"
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
chunk_left_context=40
chunk_right_context=40
dropout_schedule='0,0@0.20,0.15@0.50,0'
label_delay=0
num_epochs=6

common_egs_dir=
xent_regularize=0.1

# training options
srand=0
remove_egs=true
reporting_email=

#decode options
extra_right_context=40
extra_left_context=40
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
gmm_dir=/yrfs4/asr/yhtu3/chime6/asr/s5/exp/tri3_cleaned
ali_dir=/yrfs4/asr/lichai2/CHiME6/s5_track1/exp/tri3_cleaned_ali_train_worn_gss_BFmultiANDseletedarray_comb2s_cleaned_sp
tree_dir=/yrfs4/asr/lichai2/CHiME6/s5_track1/exp/chain_train_worn_gss_BFmultiANDseletedarray_comb2s_cleaned_rvb/tree_sp
lang=/yrfs4/asr/yhtu3/chime6/asr/s5/data/lang_chain
lat_dir=/yrfs4/asr/lichai2/CHiME6/s5_track1/exp/chain_train_worn_gss_BFmultiANDseletedarray_comb2s_cleaned_rvb/tri3_cleaned_train_worn_gss_BFmultiANDseletedarray_comb2s_cleaned_sp_lats
dir=exp/chain_train_worn_gss_BFmultiANDseletedarray_comb2s_cleaned_rvb/ResnetDyliu_tdnn_Resbilstm_projectlayer512_SpecAugment_Epoch6_addLinearComponent_rmMFCC
train_data_dir=/yrfs4/asr/lichai2/CHiME6/s5_track1/data/train_worn_gss_BFmultiANDseletedarray_combs2s_cleaned_sp_hires
lores_train_data_dir=/yrfs4/asr/lichai2/CHiME6/s5_track1/data/train_worn_gss_BFmultiANDseletedarray_combs2s_cleaned_sp
train_ivector_dir=exp/nnet3_train_worn_gss_comb2s_cleaned_rvb/ivectors_train_worn_gss_BFmultiANDseletedarray_combs2s_cleaned_sp_hires


if [ $label_delay -gt 0 ]; then dir=${dir}_ld$label_delay; fi

xent_regularize=0.1


if [ $stage -le 13 ]; then
  echo "$0: creating neural net configs using the xconfig parser";
  mkdir -p $dir
  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
  lstm_opts="decay-time=20"
  nf1=64
  nf2=128
  nf3=256
  nb3=128
  res_opts="bypass-source=batchnorm allow-zero-padding=false l2-regularize=0.01"
  cnn_opts="l2-regularize=0.01"
  ivector_affine_opts="l2-regularize=0.01"
  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  
  idct-layer name=idct input=input dim=40 cepstral-lifter=22 affine-transform-file=$dir/configs/idct.mat
  linear-component name=ivector-linear $ivector_affine_opts dim=200 input=ReplaceIndex(ivector, t, 0)
  batchnorm-component name=ivector-batchnorm target-rms=0.025
  batchnorm-component name=idct-batchnorm input=idct

  spec-augment-layer name=idct-spec-augment freq-max-proportion=0.5 time-zeroed-proportion=0.2 time-mask-max-frames=20
  combine-feature-maps-layer name=combine_inputs input=Append(idct-spec-augment, ivector-batchnorm) num-filters1=1 num-filters2=5 height=40
  
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

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=1024
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1) dim=1024
  relu-batchnorm-layer name=tdnn3 input=Append(-1,0,1) dim=1024

  # check steps/libs/nnet3/xconfig/lstm.py for the other options and defaults
  fast-lstmp-layer name=blstm1-forward input=tdnn3 cell-dim=1024 recurrent-projection-dim=512 non-recurrent-projection-dim=512 delay=-3 dropout-proportion=0.0 $lstm_opts
  fast-lstmp-layer name=blstm1-backward input=blstm1-forward cell-dim=1024 recurrent-projection-dim=512 non-recurrent-projection-dim=512 delay=3 dropout-proportion=0.0 $lstm_opts
  

  relu-batchnorm-layer name=tdnn4 input=Append(Offset(Append(blstm1-forward,blstm1-backward),-3),Append(blstm1-forward,blstm1-backward),Offset(Append(blstm1-forward,blstm1-backward),3)) dim=1024
  relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=1024
  relu-batchnorm-layer name=tdnn6 input=Append(-3,0,3) dim=1024
  fast-lstmp-layer name=blstm2-forward input=tdnn6 cell-dim=1024 recurrent-projection-dim=512 non-recurrent-projection-dim=512 delay=-3 dropout-proportion=0.0 $lstm_opts
  fast-lstmp-layer name=blstm2-backward input=blstm2-forward cell-dim=1024 recurrent-projection-dim=512 non-recurrent-projection-dim=512 delay=3 dropout-proportion=0.0 $lstm_opts
  
  relu-batchnorm-layer name=tdnn7 input=Append(Offset(Append(blstm2-forward,blstm2-backward),-3),Append(blstm2-forward,blstm2-backward),Offset(Append(blstm2-forward,blstm2-backward),3)) dim=1024
  relu-batchnorm-layer name=tdnn8 input=Append(-3,0,3) dim=1024
  relu-batchnorm-layer name=tdnn9 input=Append(-3,0,3) dim=1024
  fast-lstmp-layer name=blstm3-forward input=tdnn9 cell-dim=1024 recurrent-projection-dim=512 non-recurrent-projection-dim=512 delay=-3 dropout-proportion=0.0 $lstm_opts
  fast-lstmp-layer name=blstm3-backward input=blstm3-forward cell-dim=1024 recurrent-projection-dim=512 non-recurrent-projection-dim=512 delay=3 dropout-proportion=0.0 $lstm_opts

  linear-component name=prefinal-l input=Append(blstm3-forward,blstm3-backward) dim=256 $linear_opts

  ## adding the layers for chain branch
  output-layer name=output input=prefinal-l output-delay=$label_delay include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  output-layer name=output-xent input=prefinal-l output-delay=$label_delay dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5

EOF

  exp/script_dyliu2/steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 14 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/ami-$(date +'%m_%d_%H_%M')/s5b/$dir/egs/storage $dir/egs/storage
  fi

 steps/nnet3/chain/train.py --stage -1 \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $chunk_width \
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context $chunk_right_context \
    --egs.chunk-left-context-initial 0 \
    --egs.chunk-right-context-final 0 \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.num-chunk-per-minibatch 64,32 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.proportional-shrink 5 \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 4 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.max-param-change 2.0 \
    --trainer.deriv-truncate-margin 8 \
    --cleanup.remove-egs false \
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir
fi

if [ $stage -le 16 ]; then
  # First the options that are passed through to run_ivector_common.sh
  # (some of which are also used in this script directly).

  # The rest are configs specific to this script.  Most of the parameters
  # are just hardcoded at this level, in the commands below.
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
  for data in $test_sets2; do
    (
      local/nnet3/decode.sh \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --stage 0\
        --extra-left-context 40 \
        --extra-right-context 40\
        --extra-left-context-initial 0 \
        --extra-right-context-final 0 \
        --frames-per-chunk 150 --nj 123 \
        --ivector-dir exp/nnet3_train_worn_gss_comb2s_cleaned_rvb \
        data/${data} /work/asr/yhtu3/asr/s5/s5_train_worn_gss/data/lang $tree_dir/graph $dir
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

