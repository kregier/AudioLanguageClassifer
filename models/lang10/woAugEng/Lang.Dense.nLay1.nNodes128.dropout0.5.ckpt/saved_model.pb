î
Í£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.12v2.3.0-54-gfcc4b966f18çÓ
z
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	
*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
t
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nametrue_positives
m
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes
:*
dtype0
v
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_positives
o
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes
:*
dtype0
x
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_1
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:*
dtype0
v
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_negatives
o
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes
:*
dtype0

Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
x
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes
:	
*
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:*
dtype0

Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
x
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes
:	
*
dtype0
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
&
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ñ%
valueÇ%BÄ% B½%
Ù
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api

iter

 beta_1

!beta_2
	"decay
#learning_ratemTmUmVmWvXvYvZv[

0
1
2
3

0
1
2
3
 
­
	variables
$layer_metrics
%layer_regularization_losses
trainable_variables
&metrics

'layers
regularization_losses
(non_trainable_variables
 
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
	variables
)layer_metrics
*layer_regularization_losses
+metrics
trainable_variables

,layers
regularization_losses
-non_trainable_variables
 
 
 
­
	variables
.layer_metrics
/layer_regularization_losses
0metrics
trainable_variables

1layers
regularization_losses
2non_trainable_variables
 
 
 
­
	variables
3layer_metrics
4layer_regularization_losses
5metrics
trainable_variables

6layers
regularization_losses
7non_trainable_variables
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
	variables
8layer_metrics
9layer_regularization_losses
:metrics
trainable_variables

;layers
regularization_losses
<non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 

=0
>1
?2
@3

0
1
2
3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Atotal
	Bcount
C	variables
D	keras_api
D
	Etotal
	Fcount
G
_fn_kwargs
H	variables
I	keras_api
W
J
thresholds
Ktrue_positives
Lfalse_positives
M	variables
N	keras_api
W
O
thresholds
Ptrue_positives
Qfalse_negatives
R	variables
S	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

A0
B1

C	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

E0
F1

H	variables
 
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE

K0
L1

M	variables
 
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

P0
Q1

R	variables
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
r
serving_default_input_2Placeholder*#
_output_shapes
: 
*
dtype0*
shape: 

ð
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2dense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_signature_wrapper_192787514
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
À	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOpConst*&
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_save_192787860
Ç
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_2/kerneldense_2/biasdense_3/kerneldense_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1true_positivesfalse_positivestrue_positives_1false_negativesAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/v*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference__traced_restore_192787945÷ì
¿

K__inference_sequential_1_layer_call_and_return_conditional_losses_192787654

inputs-
)dense_2_tensordot_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity°
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 dense_2/Tensordot/ReadVariableOp
dense_2/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@     2!
dense_2/Tensordot/Reshape/shape
dense_2/Tensordot/ReshapeReshapeinputs(dense_2/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
À2
dense_2/Tensordot/Reshape·
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
À2
dense_2/Tensordot/MatMul
dense_2/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    
      2
dense_2/Tensordot/shape¥
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0 dense_2/Tensordot/shape:output:0*
T0*#
_output_shapes
: 
2
dense_2/Tensordot¥
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
: 
2
dense_2/BiasAddl
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*#
_output_shapes
: 
2
dense_2/Relu~
dropout_1/IdentityIdentitydense_2/Relu:activations:0*
T0*#
_output_shapes
: 
2
dropout_1/Identitys
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_1/Const
flatten_1/ReshapeReshapedropout_1/Identity:output:0flatten_1/Const:output:0*
T0*
_output_shapes
:	 
2
flatten_1/Reshape¦
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
dense_3/MatMul/ReadVariableOp
dense_3/MatMulMatMulflatten_1/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
dense_3/MatMul¤
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
dense_3/BiasAddp
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*
_output_shapes

: 2
dense_3/Softmaxd
IdentityIdentitydense_3/Softmax:softmax:0*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*2
_input_shapes!
: 
:::::T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
£
£
0__inference_sequential_1_layer_call_fn_192787667

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_1_layer_call_and_return_conditional_losses_1927874512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿ
::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
·9


"__inference__traced_save_192787860
file_prefix-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop-
)savev2_true_positives_read_readvariableop.
*savev2_false_positives_read_readvariableop/
+savev2_true_positives_1_read_readvariableop.
*savev2_false_negatives_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_6675306910a74b189d9e40abb2b24481/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameÐ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*â
valueØBÕB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¼
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableop+savev2_true_positives_1_read_readvariableop*savev2_false_negatives_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *(
dtypes
2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*¯
_input_shapes
: :
::	
:: : : : : : : : : :::::
::	
::
::	
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	
: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	
: 

_output_shapes
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	
: 

_output_shapes
::

_output_shapes
: 
©

K__inference_sequential_1_layer_call_and_return_conditional_losses_192787480

inputs
dense_2_192787467
dense_2_192787469
dense_3_192787474
dense_3_192787476
identity¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall
dense_2/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_192787467dense_2_192787469*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
: 
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_2_layer_call_and_return_conditional_losses_1927873282!
dense_2/StatefulPartitionedCallø
dropout_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
: 
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_1_layer_call_and_return_conditional_losses_1927873612
dropout_1/PartitionedCallî
flatten_1/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	 
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_flatten_1_layer_call_and_return_conditional_losses_1927873802
flatten_1/PartitionedCall«
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_192787474dense_3_192787476*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_3_layer_call_and_return_conditional_losses_1927873992!
dense_3/StatefulPartitionedCall·
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*2
_input_shapes!
: 
::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ü
±
F__inference_dense_2_layer_call_and_return_conditional_losses_192787328

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype02
Tensordot/ReadVariableOp
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@     2
Tensordot/Reshape/shape
Tensordot/ReshapeReshapeinputs Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
À2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
À2
Tensordot/MatMulw
Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    
      2
Tensordot/shape
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*#
_output_shapes
: 
2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
: 
2	
BiasAddT
ReluReluBiasAdd:output:0*
T0*#
_output_shapes
: 
2
Relub
IdentityIdentityRelu:activations:0*
T0*#
_output_shapes
: 
2

Identity"
identityIdentity:output:0**
_input_shapes
: 
:::K G
#
_output_shapes
: 

 
_user_specified_nameinputs
£
£
0__inference_sequential_1_layer_call_fn_192787680

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_1_layer_call_and_return_conditional_losses_1927874802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿ
::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ù
®
K__inference_sequential_1_layer_call_and_return_conditional_losses_192787451

inputs
dense_2_192787438
dense_2_192787440
dense_3_192787445
dense_3_192787447
identity¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall
dense_2/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_192787438dense_2_192787440*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
: 
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_2_layer_call_and_return_conditional_losses_1927873282!
dense_2/StatefulPartitionedCall
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
: 
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_1_layer_call_and_return_conditional_losses_1927873562#
!dropout_1/StatefulPartitionedCallö
flatten_1/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	 
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_flatten_1_layer_call_and_return_conditional_losses_1927873802
flatten_1/PartitionedCall«
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_192787445dense_3_192787447*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_3_layer_call_and_return_conditional_losses_1927873992!
dense_3/StatefulPartitionedCallÛ
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*2
_input_shapes!
: 
::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ñ

+__inference_dense_2_layer_call_fn_192787704

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
: 
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_2_layer_call_and_return_conditional_losses_1927873282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
: 
2

Identity"
identityIdentity:output:0**
_input_shapes
: 
::22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
: 

 
_user_specified_nameinputs
»
f
H__inference_dropout_1_layer_call_and_return_conditional_losses_192787361

inputs

identity_1V
IdentityIdentityinputs*
T0*#
_output_shapes
: 
2

Identitye

Identity_1IdentityIdentity:output:0*
T0*#
_output_shapes
: 
2

Identity_1"!

identity_1Identity_1:output:0*"
_input_shapes
: 
:K G
#
_output_shapes
: 

 
_user_specified_nameinputs
»
f
H__inference_dropout_1_layer_call_and_return_conditional_losses_192787721

inputs

identity_1V
IdentityIdentityinputs*
T0*#
_output_shapes
: 
2

Identitye

Identity_1IdentityIdentity:output:0*
T0*#
_output_shapes
: 
2

Identity_1"!

identity_1Identity_1:output:0*"
_input_shapes
: 
:K G
#
_output_shapes
: 

 
_user_specified_nameinputs

I
-__inference_flatten_1_layer_call_fn_192787742

inputs
identity¾
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	 
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_flatten_1_layer_call_and_return_conditional_losses_1927873802
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	 
2

Identity"
identityIdentity:output:0*"
_input_shapes
: 
:K G
#
_output_shapes
: 

 
_user_specified_nameinputs
¦
¤
0__inference_sequential_1_layer_call_fn_192787584
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_1_layer_call_and_return_conditional_losses_1927874512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿ
::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_2

g
H__inference_dropout_1_layer_call_and_return_conditional_losses_192787356

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consto
dropout/MulMulinputsdropout/Const:output:0*
T0*#
_output_shapes
: 
2
dropout/Muls
dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"    
      2
dropout/Shape°
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*#
_output_shapes
: 
*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yº
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
: 
2
dropout/GreaterEqual{
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
: 
2
dropout/Castv
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*#
_output_shapes
: 
2
dropout/Mul_1a
IdentityIdentitydropout/Mul_1:z:0*
T0*#
_output_shapes
: 
2

Identity"
identityIdentity:output:0*"
_input_shapes
: 
:K G
#
_output_shapes
: 

 
_user_specified_nameinputs

d
H__inference_flatten_1_layer_call_and_return_conditional_losses_192787380

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Const_
ReshapeReshapeinputsConst:output:0*
T0*
_output_shapes
:	 
2	
Reshape\
IdentityIdentityReshape:output:0*
T0*
_output_shapes
:	 
2

Identity"
identityIdentity:output:0*"
_input_shapes
: 
:K G
#
_output_shapes
: 

 
_user_specified_nameinputs

I
-__inference_dropout_1_layer_call_fn_192787731

inputs
identityÂ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
: 
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_1_layer_call_and_return_conditional_losses_1927873612
PartitionedCallh
IdentityIdentityPartitionedCall:output:0*
T0*#
_output_shapes
: 
2

Identity"
identityIdentity:output:0*"
_input_shapes
: 
:K G
#
_output_shapes
: 

 
_user_specified_nameinputs
¿

+__inference_dense_3_layer_call_fn_192787762

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallí
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_3_layer_call_and_return_conditional_losses_1927873992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*&
_input_shapes
:	 
::22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	 

 
_user_specified_nameinputs
å!

K__inference_sequential_1_layer_call_and_return_conditional_losses_192787629

inputs-
)dense_2_tensordot_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity°
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 dense_2/Tensordot/ReadVariableOp
dense_2/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@     2!
dense_2/Tensordot/Reshape/shape
dense_2/Tensordot/ReshapeReshapeinputs(dense_2/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
À2
dense_2/Tensordot/Reshape·
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
À2
dense_2/Tensordot/MatMul
dense_2/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    
      2
dense_2/Tensordot/shape¥
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0 dense_2/Tensordot/shape:output:0*
T0*#
_output_shapes
: 
2
dense_2/Tensordot¥
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
: 
2
dense_2/BiasAddl
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*#
_output_shapes
: 
2
dense_2/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/dropout/Const¡
dropout_1/dropout/MulMuldense_2/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*#
_output_shapes
: 
2
dropout_1/dropout/Mul
dropout_1/dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"    
      2
dropout_1/dropout/ShapeÎ
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*#
_output_shapes
: 
*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_1/dropout/GreaterEqual/yâ
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
: 
2 
dropout_1/dropout/GreaterEqual
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
: 
2
dropout_1/dropout/Cast
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*#
_output_shapes
: 
2
dropout_1/dropout/Mul_1s
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_1/Const
flatten_1/ReshapeReshapedropout_1/dropout/Mul_1:z:0flatten_1/Const:output:0*
T0*
_output_shapes
:	 
2
flatten_1/Reshape¦
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
dense_3/MatMul/ReadVariableOp
dense_3/MatMulMatMulflatten_1/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
dense_3/MatMul¤
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
dense_3/BiasAddp
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*
_output_shapes

: 2
dense_3/Softmaxd
IdentityIdentitydense_3/Softmax:softmax:0*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*2
_input_shapes!
: 
:::::T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ü
±
F__inference_dense_2_layer_call_and_return_conditional_losses_192787695

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype02
Tensordot/ReadVariableOp
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@     2
Tensordot/Reshape/shape
Tensordot/ReshapeReshapeinputs Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
À2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
À2
Tensordot/MatMulw
Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    
      2
Tensordot/shape
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*#
_output_shapes
: 
2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
: 
2	
BiasAddT
ReluReluBiasAdd:output:0*
T0*#
_output_shapes
: 
2
Relub
IdentityIdentityRelu:activations:0*
T0*#
_output_shapes
: 
2

Identity"
identityIdentity:output:0**
_input_shapes
: 
:::K G
#
_output_shapes
: 

 
_user_specified_nameinputs

­
$__inference__wrapped_model_192787309
input_2:
6sequential_1_dense_2_tensordot_readvariableop_resource8
4sequential_1_dense_2_biasadd_readvariableop_resource7
3sequential_1_dense_3_matmul_readvariableop_resource8
4sequential_1_dense_3_biasadd_readvariableop_resource
identity×
-sequential_1/dense_2/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype02/
-sequential_1/dense_2/Tensordot/ReadVariableOp­
,sequential_1/dense_2/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@     2.
,sequential_1/dense_2/Tensordot/Reshape/shapeÆ
&sequential_1/dense_2/Tensordot/ReshapeReshapeinput_25sequential_1/dense_2/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
À2(
&sequential_1/dense_2/Tensordot/Reshapeë
%sequential_1/dense_2/Tensordot/MatMulMatMul/sequential_1/dense_2/Tensordot/Reshape:output:05sequential_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
À2'
%sequential_1/dense_2/Tensordot/MatMul¡
$sequential_1/dense_2/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    
      2&
$sequential_1/dense_2/Tensordot/shapeÙ
sequential_1/dense_2/TensordotReshape/sequential_1/dense_2/Tensordot/MatMul:product:0-sequential_1/dense_2/Tensordot/shape:output:0*
T0*#
_output_shapes
: 
2 
sequential_1/dense_2/TensordotÌ
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_1/dense_2/BiasAdd/ReadVariableOpÓ
sequential_1/dense_2/BiasAddBiasAdd'sequential_1/dense_2/Tensordot:output:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
: 
2
sequential_1/dense_2/BiasAdd
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*#
_output_shapes
: 
2
sequential_1/dense_2/Relu¥
sequential_1/dropout_1/IdentityIdentity'sequential_1/dense_2/Relu:activations:0*
T0*#
_output_shapes
: 
2!
sequential_1/dropout_1/Identity
sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
sequential_1/flatten_1/ConstÆ
sequential_1/flatten_1/ReshapeReshape(sequential_1/dropout_1/Identity:output:0%sequential_1/flatten_1/Const:output:0*
T0*
_output_shapes
:	 
2 
sequential_1/flatten_1/ReshapeÍ
*sequential_1/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_3_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02,
*sequential_1/dense_3/MatMul/ReadVariableOpÊ
sequential_1/dense_3/MatMulMatMul'sequential_1/flatten_1/Reshape:output:02sequential_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
sequential_1/dense_3/MatMulË
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_1/dense_3/BiasAdd/ReadVariableOpÌ
sequential_1/dense_3/BiasAddBiasAdd%sequential_1/dense_3/MatMul:product:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
sequential_1/dense_3/BiasAdd
sequential_1/dense_3/SoftmaxSoftmax%sequential_1/dense_3/BiasAdd:output:0*
T0*
_output_shapes

: 2
sequential_1/dense_3/Softmaxq
IdentityIdentity&sequential_1/dense_3/Softmax:softmax:0*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*2
_input_shapes!
: 
:::::U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_2

f
-__inference_dropout_1_layer_call_fn_192787726

inputs
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
: 
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_1_layer_call_and_return_conditional_losses_1927873562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
: 
2

Identity"
identityIdentity:output:0*"
_input_shapes
: 
22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
: 

 
_user_specified_nameinputs
ÿ
®
F__inference_dense_3_layer_call_and_return_conditional_losses_192787399

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2	
BiasAddX
SoftmaxSoftmaxBiasAdd:output:0*
T0*
_output_shapes

: 2	
Softmax\
IdentityIdentitySoftmax:softmax:0*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*&
_input_shapes
:	 
:::G C

_output_shapes
:	 

 
_user_specified_nameinputs
ÿ
®
F__inference_dense_3_layer_call_and_return_conditional_losses_192787753

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2	
BiasAddX
SoftmaxSoftmaxBiasAdd:output:0*
T0*
_output_shapes

: 2	
Softmax\
IdentityIdentitySoftmax:softmax:0*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*&
_input_shapes
:	 
:::G C

_output_shapes
:	 

 
_user_specified_nameinputs

g
H__inference_dropout_1_layer_call_and_return_conditional_losses_192787716

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consto
dropout/MulMulinputsdropout/Const:output:0*
T0*#
_output_shapes
: 
2
dropout/Muls
dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"    
      2
dropout/Shape°
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*#
_output_shapes
: 
*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yº
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
: 
2
dropout/GreaterEqual{
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
: 
2
dropout/Castv
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*#
_output_shapes
: 
2
dropout/Mul_1a
IdentityIdentitydropout/Mul_1:z:0*
T0*#
_output_shapes
: 
2

Identity"
identityIdentity:output:0*"
_input_shapes
: 
:K G
#
_output_shapes
: 

 
_user_specified_nameinputs
ä

'__inference_signature_wrapper_192787514
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__wrapped_model_1927873092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*2
_input_shapes!
: 
::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
: 

!
_user_specified_name	input_2
è!
 
K__inference_sequential_1_layer_call_and_return_conditional_losses_192787546
input_2-
)dense_2_tensordot_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity°
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 dense_2/Tensordot/ReadVariableOp
dense_2/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@     2!
dense_2/Tensordot/Reshape/shape
dense_2/Tensordot/ReshapeReshapeinput_2(dense_2/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
À2
dense_2/Tensordot/Reshape·
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
À2
dense_2/Tensordot/MatMul
dense_2/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    
      2
dense_2/Tensordot/shape¥
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0 dense_2/Tensordot/shape:output:0*
T0*#
_output_shapes
: 
2
dense_2/Tensordot¥
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
: 
2
dense_2/BiasAddl
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*#
_output_shapes
: 
2
dense_2/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/dropout/Const¡
dropout_1/dropout/MulMuldense_2/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*#
_output_shapes
: 
2
dropout_1/dropout/Mul
dropout_1/dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"    
      2
dropout_1/dropout/ShapeÎ
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*#
_output_shapes
: 
*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_1/dropout/GreaterEqual/yâ
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
: 
2 
dropout_1/dropout/GreaterEqual
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
: 
2
dropout_1/dropout/Cast
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*#
_output_shapes
: 
2
dropout_1/dropout/Mul_1s
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_1/Const
flatten_1/ReshapeReshapedropout_1/dropout/Mul_1:z:0flatten_1/Const:output:0*
T0*
_output_shapes
:	 
2
flatten_1/Reshape¦
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
dense_3/MatMul/ReadVariableOp
dense_3/MatMulMatMulflatten_1/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
dense_3/MatMul¤
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
dense_3/BiasAddp
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*
_output_shapes

: 2
dense_3/Softmaxd
IdentityIdentitydense_3/Softmax:softmax:0*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*2
_input_shapes!
: 
:::::U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_2
¦
¤
0__inference_sequential_1_layer_call_fn_192787597
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_1_layer_call_and_return_conditional_losses_1927874802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿ
::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_2

d
H__inference_flatten_1_layer_call_and_return_conditional_losses_192787737

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Const_
ReshapeReshapeinputsConst:output:0*
T0*
_output_shapes
:	 
2	
Reshape\
IdentityIdentityReshape:output:0*
T0*
_output_shapes
:	 
2

Identity"
identityIdentity:output:0*"
_input_shapes
: 
:K G
#
_output_shapes
: 

 
_user_specified_nameinputs
Â
 
K__inference_sequential_1_layer_call_and_return_conditional_losses_192787571
input_2-
)dense_2_tensordot_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity°
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 dense_2/Tensordot/ReadVariableOp
dense_2/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@     2!
dense_2/Tensordot/Reshape/shape
dense_2/Tensordot/ReshapeReshapeinput_2(dense_2/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
À2
dense_2/Tensordot/Reshape·
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
À2
dense_2/Tensordot/MatMul
dense_2/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    
      2
dense_2/Tensordot/shape¥
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0 dense_2/Tensordot/shape:output:0*
T0*#
_output_shapes
: 
2
dense_2/Tensordot¥
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
: 
2
dense_2/BiasAddl
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*#
_output_shapes
: 
2
dense_2/Relu~
dropout_1/IdentityIdentitydense_2/Relu:activations:0*
T0*#
_output_shapes
: 
2
dropout_1/Identitys
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_1/Const
flatten_1/ReshapeReshapedropout_1/Identity:output:0flatten_1/Const:output:0*
T0*
_output_shapes
:	 
2
flatten_1/Reshape¦
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
dense_3/MatMul/ReadVariableOp
dense_3/MatMulMatMulflatten_1/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
dense_3/MatMul¤
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
dense_3/BiasAddp
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*
_output_shapes

: 2
dense_3/Softmaxd
IdentityIdentitydense_3/Softmax:softmax:0*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*2
_input_shapes!
: 
:::::U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_2
õi
Ç
%__inference__traced_restore_192787945
file_prefix#
assignvariableop_dense_2_kernel#
assignvariableop_1_dense_2_bias%
!assignvariableop_2_dense_3_kernel#
assignvariableop_3_dense_3_bias 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rate
assignvariableop_9_total
assignvariableop_10_count
assignvariableop_11_total_1
assignvariableop_12_count_1&
"assignvariableop_13_true_positives'
#assignvariableop_14_false_positives(
$assignvariableop_15_true_positives_1'
#assignvariableop_16_false_negatives-
)assignvariableop_17_adam_dense_2_kernel_m+
'assignvariableop_18_adam_dense_2_bias_m-
)assignvariableop_19_adam_dense_3_kernel_m+
'assignvariableop_20_adam_dense_3_bias_m-
)assignvariableop_21_adam_dense_2_kernel_v+
'assignvariableop_22_adam_dense_2_bias_v-
)assignvariableop_23_adam_dense_3_kernel_v+
'assignvariableop_24_adam_dense_3_bias_v
identity_26¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ö
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*â
valueØBÕB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÂ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices­
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¤
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¦
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¤
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4¡
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5£
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6£
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¢
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8ª
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¡
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11£
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12£
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13ª
AssignVariableOp_13AssignVariableOp"assignvariableop_13_true_positivesIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14«
AssignVariableOp_14AssignVariableOp#assignvariableop_14_false_positivesIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¬
AssignVariableOp_15AssignVariableOp$assignvariableop_15_true_positives_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16«
AssignVariableOp_16AssignVariableOp#assignvariableop_16_false_negativesIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17±
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_2_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¯
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_2_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19±
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_dense_3_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¯
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_3_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21±
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_2_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¯
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_2_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23±
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_3_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¯
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_3_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_249
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_25÷
Identity_26IdentityIdentity_25:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_26"#
identity_26Identity_26:output:0*y
_input_shapesh
f: :::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_default
7
input_2,
serving_default_input_2:0 
2
dense_3'
StatefulPartitionedCall:0 tensorflow/serving/predict:

layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
\_default_save_signature
]__call__
*^&call_and_return_all_conditional_losses"Û
_tf_keras_sequential¼{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [32, 10, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 11, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 10, 128]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [32, 10, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 11, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy", "Precision", "Recall"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
õ

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
___call__
*`&call_and_return_all_conditional_losses"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 10, 128]}}
å
	variables
trainable_variables
regularization_losses
	keras_api
a__call__
*b&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
æ
	variables
trainable_variables
regularization_losses
	keras_api
c__call__
*d&call_and_return_all_conditional_losses"×
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
õ

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
e__call__
*f&call_and_return_all_conditional_losses"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 11, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1280}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 1280]}}

iter

 beta_1

!beta_2
	"decay
#learning_ratemTmUmVmWvXvYvZv["
	optimizer
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
	variables
$layer_metrics
%layer_regularization_losses
trainable_variables
&metrics

'layers
regularization_losses
(non_trainable_variables
]__call__
\_default_save_signature
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
,
gserving_default"
signature_map
": 
2dense_2/kernel
:2dense_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
	variables
)layer_metrics
*layer_regularization_losses
+metrics
trainable_variables

,layers
regularization_losses
-non_trainable_variables
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
	variables
.layer_metrics
/layer_regularization_losses
0metrics
trainable_variables

1layers
regularization_losses
2non_trainable_variables
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
	variables
3layer_metrics
4layer_regularization_losses
5metrics
trainable_variables

6layers
regularization_losses
7non_trainable_variables
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
!:	
2dense_3/kernel
:2dense_3/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
	variables
8layer_metrics
9layer_regularization_losses
:metrics
trainable_variables

;layers
regularization_losses
<non_trainable_variables
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
=0
>1
?2
@3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
»
	Atotal
	Bcount
C	variables
D	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
ÿ
	Etotal
	Fcount
G
_fn_kwargs
H	variables
I	keras_api"¸
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
£
J
thresholds
Ktrue_positives
Lfalse_positives
M	variables
N	keras_api"É
_tf_keras_metric®{"class_name": "Precision", "name": "precision", "dtype": "float32", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}

O
thresholds
Ptrue_positives
Qfalse_negatives
R	variables
S	keras_api"À
_tf_keras_metric¥{"class_name": "Recall", "name": "recall", "dtype": "float32", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
:  (2total
:  (2count
.
A0
B1"
trackable_list_wrapper
-
C	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
E0
F1"
trackable_list_wrapper
-
H	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
.
K0
L1"
trackable_list_wrapper
-
M	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
.
P0
Q1"
trackable_list_wrapper
-
R	variables"
_generic_user_object
':%
2Adam/dense_2/kernel/m
 :2Adam/dense_2/bias/m
&:$	
2Adam/dense_3/kernel/m
:2Adam/dense_3/bias/m
':%
2Adam/dense_2/kernel/v
 :2Adam/dense_2/bias/v
&:$	
2Adam/dense_3/kernel/v
:2Adam/dense_3/bias/v
ç2ä
$__inference__wrapped_model_192787309»
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *+¢(
&#
input_2ÿÿÿÿÿÿÿÿÿ

2
0__inference_sequential_1_layer_call_fn_192787667
0__inference_sequential_1_layer_call_fn_192787680
0__inference_sequential_1_layer_call_fn_192787584
0__inference_sequential_1_layer_call_fn_192787597À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ú2÷
K__inference_sequential_1_layer_call_and_return_conditional_losses_192787571
K__inference_sequential_1_layer_call_and_return_conditional_losses_192787654
K__inference_sequential_1_layer_call_and_return_conditional_losses_192787546
K__inference_sequential_1_layer_call_and_return_conditional_losses_192787629À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Õ2Ò
+__inference_dense_2_layer_call_fn_192787704¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_dense_2_layer_call_and_return_conditional_losses_192787695¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
-__inference_dropout_1_layer_call_fn_192787726
-__inference_dropout_1_layer_call_fn_192787731´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Î2Ë
H__inference_dropout_1_layer_call_and_return_conditional_losses_192787716
H__inference_dropout_1_layer_call_and_return_conditional_losses_192787721´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
×2Ô
-__inference_flatten_1_layer_call_fn_192787742¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_flatten_1_layer_call_and_return_conditional_losses_192787737¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_dense_3_layer_call_fn_192787762¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_dense_3_layer_call_and_return_conditional_losses_192787753¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
6B4
'__inference_signature_wrapper_192787514input_2
$__inference__wrapped_model_192787309g5¢2
+¢(
&#
input_2ÿÿÿÿÿÿÿÿÿ

ª "(ª%
#
dense_3
dense_3 
F__inference_dense_2_layer_call_and_return_conditional_losses_192787695T+¢(
!¢

inputs 

ª "!¢

0 

 v
+__inference_dense_2_layer_call_fn_192787704G+¢(
!¢

inputs 

ª " 

F__inference_dense_3_layer_call_and_return_conditional_losses_192787753K'¢$
¢

inputs	 

ª "¢

0 
 m
+__inference_dense_3_layer_call_fn_192787762>'¢$
¢

inputs	 

ª "  
H__inference_dropout_1_layer_call_and_return_conditional_losses_192787716T/¢,
%¢"

inputs 

p
ª "!¢

0 

  
H__inference_dropout_1_layer_call_and_return_conditional_losses_192787721T/¢,
%¢"

inputs 

p 
ª "!¢

0 

 x
-__inference_dropout_1_layer_call_fn_192787726G/¢,
%¢"

inputs 

p
ª " 
x
-__inference_dropout_1_layer_call_fn_192787731G/¢,
%¢"

inputs 

p 
ª " 

H__inference_flatten_1_layer_call_and_return_conditional_losses_192787737L+¢(
!¢

inputs 

ª "¢

0	 

 p
-__inference_flatten_1_layer_call_fn_192787742?+¢(
!¢

inputs 

ª "	 
²
K__inference_sequential_1_layer_call_and_return_conditional_losses_192787546c=¢:
3¢0
&#
input_2ÿÿÿÿÿÿÿÿÿ

p

 
ª "¢

0 
 ²
K__inference_sequential_1_layer_call_and_return_conditional_losses_192787571c=¢:
3¢0
&#
input_2ÿÿÿÿÿÿÿÿÿ

p 

 
ª "¢

0 
 ±
K__inference_sequential_1_layer_call_and_return_conditional_losses_192787629b<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿ

p

 
ª "¢

0 
 ±
K__inference_sequential_1_layer_call_and_return_conditional_losses_192787654b<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿ

p 

 
ª "¢

0 
 
0__inference_sequential_1_layer_call_fn_192787584V=¢:
3¢0
&#
input_2ÿÿÿÿÿÿÿÿÿ

p

 
ª " 
0__inference_sequential_1_layer_call_fn_192787597V=¢:
3¢0
&#
input_2ÿÿÿÿÿÿÿÿÿ

p 

 
ª " 
0__inference_sequential_1_layer_call_fn_192787667U<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿ

p

 
ª " 
0__inference_sequential_1_layer_call_fn_192787680U<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿ

p 

 
ª " 
'__inference_signature_wrapper_192787514i7¢4
¢ 
-ª*
(
input_2
input_2 
"(ª%
#
dense_3
dense_3 