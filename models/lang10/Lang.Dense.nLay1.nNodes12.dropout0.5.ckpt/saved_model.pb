ňä
ÍŁ
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
ž
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
 "serve*2.3.12v2.3.0-54-gfcc4b966f18ÉË
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:x*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
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

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:x*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:x*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
&
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ĺ%
valueť%B¸% Bą%
Ů
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
$metrics
	variables
trainable_variables
regularization_losses

%layers
&layer_metrics
'non_trainable_variables
(layer_regularization_losses
 
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
)metrics
	variables
trainable_variables
regularization_losses

*layers
+layer_metrics
,non_trainable_variables
-layer_regularization_losses
 
 
 
­
.metrics
	variables
trainable_variables
regularization_losses

/layers
0layer_metrics
1non_trainable_variables
2layer_regularization_losses
 
 
 
­
3metrics
	variables
trainable_variables
regularization_losses

4layers
5layer_metrics
6non_trainable_variables
7layer_regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
8metrics
	variables
trainable_variables
regularization_losses

9layers
:layer_metrics
;non_trainable_variables
<layer_regularization_losses
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
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
r
serving_default_input_1Placeholder*#
_output_shapes
: 
*
dtype0*
shape: 

ě
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/biasdense_1/kerneldense_1/bias*
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
'__inference_signature_wrapper_152977188
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
´	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpConst*&
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
"__inference__traced_save_152977534
ť
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1true_positivesfalse_positivestrue_positives_1false_negativesAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v*%
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
%__inference__traced_restore_152977619Çĺ
ä

I__inference_sequential_layer_call_and_return_conditional_losses_152977245

inputs+
'dense_tensordot_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityŠ
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype02 
dense/Tensordot/ReadVariableOp
dense/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@     2
dense/Tensordot/Reshape/shape
dense/Tensordot/ReshapeReshapeinputs&dense/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
Ŕ2
dense/Tensordot/ReshapeŽ
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ŕ2
dense/Tensordot/MatMul
dense/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    
      2
dense/Tensordot/shape
dense/TensordotReshape dense/Tensordot/MatMul:product:0dense/Tensordot/shape:output:0*
T0*"
_output_shapes
: 
2
dense/Tensordot
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
: 
2
dense/BiasAdde

dense/ReluReludense/BiasAdd:output:0*
T0*"
_output_shapes
: 
2

dense/Reluw
dropout/IdentityIdentitydense/Relu:activations:0*
T0*"
_output_shapes
: 
2
dropout/Identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙x   2
flatten/Const
flatten/ReshapeReshapedropout/Identity:output:0flatten/Const:output:0*
T0*
_output_shapes

: x2
flatten/ReshapeĽ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:x*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
dense_1/BiasAddp
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*
_output_shapes

: 2
dense_1/Softmaxd
IdentityIdentitydense_1/Softmax:softmax:0*
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
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
Ü 

I__inference_sequential_layer_call_and_return_conditional_losses_152977220

inputs+
'dense_tensordot_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityŠ
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype02 
dense/Tensordot/ReadVariableOp
dense/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@     2
dense/Tensordot/Reshape/shape
dense/Tensordot/ReshapeReshapeinputs&dense/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
Ŕ2
dense/Tensordot/ReshapeŽ
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ŕ2
dense/Tensordot/MatMul
dense/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    
      2
dense/Tensordot/shape
dense/TensordotReshape dense/Tensordot/MatMul:product:0dense/Tensordot/shape:output:0*
T0*"
_output_shapes
: 
2
dense/Tensordot
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
: 
2
dense/BiasAdde

dense/ReluReludense/BiasAdd:output:0*
T0*"
_output_shapes
: 
2

dense/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*"
_output_shapes
: 
2
dropout/dropout/Mul
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"    
      2
dropout/dropout/ShapeÇ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*"
_output_shapes
: 
*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/yŮ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*"
_output_shapes
: 
2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*"
_output_shapes
: 
2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*"
_output_shapes
: 
2
dropout/dropout/Mul_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙x   2
flatten/Const
flatten/ReshapeReshapedropout/dropout/Mul_1:z:0flatten/Const:output:0*
T0*
_output_shapes

: x2
flatten/ReshapeĽ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:x*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
dense_1/BiasAddp
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*
_output_shapes

: 2
dense_1/Softmaxd
IdentityIdentitydense_1/Softmax:softmax:0*
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
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
ü
Ž
F__inference_dense_1_layer_call_and_return_conditional_losses_152977073

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x*
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
identityIdentity:output:0*%
_input_shapes
: x:::F B

_output_shapes

: x
 
_user_specified_nameinputs

b
F__inference_flatten_layer_call_and_return_conditional_losses_152977411

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙x   2
Const^
ReshapeReshapeinputsConst:output:0*
T0*
_output_shapes

: x2	
Reshape[
IdentityIdentityReshape:output:0*
T0*
_output_shapes

: x2

Identity"
identityIdentity:output:0*!
_input_shapes
: 
:J F
"
_output_shapes
: 

 
_user_specified_nameinputs
ß 

I__inference_sequential_layer_call_and_return_conditional_losses_152977303
input_1+
'dense_tensordot_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityŠ
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype02 
dense/Tensordot/ReadVariableOp
dense/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@     2
dense/Tensordot/Reshape/shape
dense/Tensordot/ReshapeReshapeinput_1&dense/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
Ŕ2
dense/Tensordot/ReshapeŽ
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ŕ2
dense/Tensordot/MatMul
dense/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    
      2
dense/Tensordot/shape
dense/TensordotReshape dense/Tensordot/MatMul:product:0dense/Tensordot/shape:output:0*
T0*"
_output_shapes
: 
2
dense/Tensordot
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
: 
2
dense/BiasAdde

dense/ReluReludense/BiasAdd:output:0*
T0*"
_output_shapes
: 
2

dense/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*"
_output_shapes
: 
2
dropout/dropout/Mul
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"    
      2
dropout/dropout/ShapeÇ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*"
_output_shapes
: 
*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/yŮ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*"
_output_shapes
: 
2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*"
_output_shapes
: 
2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*"
_output_shapes
: 
2
dropout/dropout/Mul_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙x   2
flatten/Const
flatten/ReshapeReshapedropout/dropout/Mul_1:z:0flatten/Const:output:0*
T0*
_output_shapes

: x2
flatten/ReshapeĽ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:x*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
dense_1/BiasAddp
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*
_output_shapes

: 2
dense_1/Softmaxd
IdentityIdentitydense_1/Softmax:softmax:0*
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
:˙˙˙˙˙˙˙˙˙

!
_user_specified_name	input_1
˝

+__inference_dense_1_layer_call_fn_152977436

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallí
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
F__inference_dense_1_layer_call_and_return_conditional_losses_1529770732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*%
_input_shapes
: x::22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

: x
 
_user_specified_nameinputs
ü

I__inference_sequential_layer_call_and_return_conditional_losses_152977154

inputs
dense_152977141
dense_152977143
dense_1_152977148
dense_1_152977150
identity˘dense/StatefulPartitionedCall˘dense_1/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_152977141dense_152977143*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
: 
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_1529770022
dense/StatefulPartitionedCallď
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
: 
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_layer_call_and_return_conditional_losses_1529770352
dropout/PartitionedCallĺ
flatten/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: x* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_layer_call_and_return_conditional_losses_1529770542
flatten/PartitionedCallŠ
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_152977148dense_1_152977150*
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
F__inference_dense_1_layer_call_and_return_conditional_losses_1529770732!
dense_1/StatefulPartitionedCallľ
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*2
_input_shapes!
: 
::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
Ó
Ż
D__inference_dense_layer_call_and_return_conditional_losses_152977002

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	*
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
Ŕ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ŕ2
Tensordot/MatMulw
Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    
      2
Tensordot/shape
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*"
_output_shapes
: 
2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp~
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
: 
2	
BiasAddS
ReluReluBiasAdd:output:0*
T0*"
_output_shapes
: 
2
Relua
IdentityIdentityRelu:activations:0*
T0*"
_output_shapes
: 
2

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
˘
˘
.__inference_sequential_layer_call_fn_152977341
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
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
GPU 2J 8 *R
fMRK
I__inference_sequential_layer_call_and_return_conditional_losses_1529771252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙
::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:˙˙˙˙˙˙˙˙˙

!
_user_specified_name	input_1

Ą
.__inference_sequential_layer_call_fn_152977258

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCall
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
GPU 2J 8 *R
fMRK
I__inference_sequential_layer_call_and_return_conditional_losses_1529771252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙
::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
Ę
~
)__inference_dense_layer_call_fn_152977378

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallď
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
: 
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_1529770022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*"
_output_shapes
: 
2

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
ů

e
F__inference_dropout_layer_call_and_return_conditional_losses_152977390

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constn
dropout/MulMulinputsdropout/Const:output:0*
T0*"
_output_shapes
: 
2
dropout/Muls
dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"    
      2
dropout/ShapeŻ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*"
_output_shapes
: 
*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yš
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*"
_output_shapes
: 
2
dropout/GreaterEqualz
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*"
_output_shapes
: 
2
dropout/Castu
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*"
_output_shapes
: 
2
dropout/Mul_1`
IdentityIdentitydropout/Mul_1:z:0*
T0*"
_output_shapes
: 
2

Identity"
identityIdentity:output:0*!
_input_shapes
: 
:J F
"
_output_shapes
: 

 
_user_specified_nameinputs

b
F__inference_flatten_layer_call_and_return_conditional_losses_152977054

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙x   2
Const^
ReshapeReshapeinputsConst:output:0*
T0*
_output_shapes

: x2	
Reshape[
IdentityIdentityReshape:output:0*
T0*
_output_shapes

: x2

Identity"
identityIdentity:output:0*!
_input_shapes
: 
:J F
"
_output_shapes
: 

 
_user_specified_nameinputs
˘
˘
.__inference_sequential_layer_call_fn_152977354
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
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
GPU 2J 8 *R
fMRK
I__inference_sequential_layer_call_and_return_conditional_losses_1529771542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙
::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:˙˙˙˙˙˙˙˙˙

!
_user_specified_name	input_1
ľ
d
F__inference_dropout_layer_call_and_return_conditional_losses_152977395

inputs

identity_1U
IdentityIdentityinputs*
T0*"
_output_shapes
: 
2

Identityd

Identity_1IdentityIdentity:output:0*
T0*"
_output_shapes
: 
2

Identity_1"!

identity_1Identity_1:output:0*!
_input_shapes
: 
:J F
"
_output_shapes
: 

 
_user_specified_nameinputs

Ą
.__inference_sequential_layer_call_fn_152977271

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCall
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
GPU 2J 8 *R
fMRK
I__inference_sequential_layer_call_and_return_conditional_losses_1529771542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙
::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
Ď
Ą
$__inference__wrapped_model_152976983
input_16
2sequential_dense_tensordot_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource5
1sequential_dense_1_matmul_readvariableop_resource6
2sequential_dense_1_biasadd_readvariableop_resource
identityĘ
)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype02+
)sequential/dense/Tensordot/ReadVariableOpĽ
(sequential/dense/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@     2*
(sequential/dense/Tensordot/Reshape/shapeş
"sequential/dense/Tensordot/ReshapeReshapeinput_11sequential/dense/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
Ŕ2$
"sequential/dense/Tensordot/ReshapeÚ
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ŕ2#
!sequential/dense/Tensordot/MatMul
 sequential/dense/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    
      2"
 sequential/dense/Tensordot/shapeČ
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0)sequential/dense/Tensordot/shape:output:0*
T0*"
_output_shapes
: 
2
sequential/dense/Tensordotż
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOpÂ
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
: 
2
sequential/dense/BiasAdd
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*"
_output_shapes
: 
2
sequential/dense/Relu
sequential/dropout/IdentityIdentity#sequential/dense/Relu:activations:0*
T0*"
_output_shapes
: 
2
sequential/dropout/Identity
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙x   2
sequential/flatten/Constľ
sequential/flatten/ReshapeReshape$sequential/dropout/Identity:output:0!sequential/flatten/Const:output:0*
T0*
_output_shapes

: x2
sequential/flatten/ReshapeĆ
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:x*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOpŔ
sequential/dense_1/MatMulMatMul#sequential/flatten/Reshape:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
sequential/dense_1/MatMulĹ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOpÄ
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
sequential/dense_1/BiasAdd
sequential/dense_1/SoftmaxSoftmax#sequential/dense_1/BiasAdd:output:0*
T0*
_output_shapes

: 2
sequential/dense_1/Softmaxo
IdentityIdentity$sequential/dense_1/Softmax:softmax:0*
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
:˙˙˙˙˙˙˙˙˙

!
_user_specified_name	input_1
ä

'__inference_signature_wrapper_152977188
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCallć
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
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
$__inference__wrapped_model_1529769832
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
_user_specified_name	input_1
ľ
d
F__inference_dropout_layer_call_and_return_conditional_losses_152977035

inputs

identity_1U
IdentityIdentityinputs*
T0*"
_output_shapes
: 
2

Identityd

Identity_1IdentityIdentity:output:0*
T0*"
_output_shapes
: 
2

Identity_1"!

identity_1Identity_1:output:0*!
_input_shapes
: 
:J F
"
_output_shapes
: 

 
_user_specified_nameinputs
ů

e
F__inference_dropout_layer_call_and_return_conditional_losses_152977030

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constn
dropout/MulMulinputsdropout/Const:output:0*
T0*"
_output_shapes
: 
2
dropout/Muls
dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"    
      2
dropout/ShapeŻ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*"
_output_shapes
: 
*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yš
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*"
_output_shapes
: 
2
dropout/GreaterEqualz
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*"
_output_shapes
: 
2
dropout/Castu
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*"
_output_shapes
: 
2
dropout/Mul_1`
IdentityIdentitydropout/Mul_1:z:0*
T0*"
_output_shapes
: 
2

Identity"
identityIdentity:output:0*!
_input_shapes
: 
:J F
"
_output_shapes
: 

 
_user_specified_nameinputs
ü
Ž
F__inference_dense_1_layer_call_and_return_conditional_losses_152977427

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x*
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
identityIdentity:output:0*%
_input_shapes
: x:::F B

_output_shapes

: x
 
_user_specified_nameinputs
Ó
Ż
D__inference_dense_layer_call_and_return_conditional_losses_152977369

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	*
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
Ŕ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ŕ2
Tensordot/MatMulw
Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    
      2
Tensordot/shape
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*"
_output_shapes
: 
2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp~
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
: 
2	
BiasAddS
ReluReluBiasAdd:output:0*
T0*"
_output_shapes
: 
2
Relua
IdentityIdentityRelu:activations:0*
T0*"
_output_shapes
: 
2

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
Ýi
ť
%__inference__traced_restore_152977619
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias%
!assignvariableop_2_dense_1_kernel#
assignvariableop_3_dense_1_bias 
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
#assignvariableop_16_false_negatives+
'assignvariableop_17_adam_dense_kernel_m)
%assignvariableop_18_adam_dense_bias_m-
)assignvariableop_19_adam_dense_1_kernel_m+
'assignvariableop_20_adam_dense_1_bias_m+
'assignvariableop_21_adam_dense_kernel_v)
%assignvariableop_22_adam_dense_bias_v-
)assignvariableop_23_adam_dense_1_kernel_v+
'assignvariableop_24_adam_dense_1_bias_v
identity_26˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_10˘AssignVariableOp_11˘AssignVariableOp_12˘AssignVariableOp_13˘AssignVariableOp_14˘AssignVariableOp_15˘AssignVariableOp_16˘AssignVariableOp_17˘AssignVariableOp_18˘AssignVariableOp_19˘AssignVariableOp_2˘AssignVariableOp_20˘AssignVariableOp_21˘AssignVariableOp_22˘AssignVariableOp_23˘AssignVariableOp_24˘AssignVariableOp_3˘AssignVariableOp_4˘AssignVariableOp_5˘AssignVariableOp_6˘AssignVariableOp_7˘AssignVariableOp_8˘AssignVariableOp_9Ö
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*â
valueŘBŐB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1˘
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ś
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¤
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4Ą
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ł
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Ł
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7˘
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ş
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
Identity_10Ą
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ł
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ł
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ş
AssignVariableOp_13AssignVariableOp"assignvariableop_13_true_positivesIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ť
AssignVariableOp_14AssignVariableOp#assignvariableop_14_false_positivesIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ź
AssignVariableOp_15AssignVariableOp$assignvariableop_15_true_positives_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ť
AssignVariableOp_16AssignVariableOp#assignvariableop_16_false_negativesIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ż
AssignVariableOp_17AssignVariableOp'assignvariableop_17_adam_dense_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18­
AssignVariableOp_18AssignVariableOp%assignvariableop_18_adam_dense_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19ą
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_dense_1_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ż
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_1_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ż
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_dense_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22­
AssignVariableOp_22AssignVariableOp%assignvariableop_22_adam_dense_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23ą
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_1_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ż
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_1_bias_vIdentity_24:output:0"/device:CPU:0*
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
_user_specified_namefile_prefix

G
+__inference_dropout_layer_call_fn_152977405

inputs
identityż
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
: 
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_layer_call_and_return_conditional_losses_1529770352
PartitionedCallg
IdentityIdentityPartitionedCall:output:0*
T0*"
_output_shapes
: 
2

Identity"
identityIdentity:output:0*!
_input_shapes
: 
:J F
"
_output_shapes
: 

 
_user_specified_nameinputs
ç

I__inference_sequential_layer_call_and_return_conditional_losses_152977328
input_1+
'dense_tensordot_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityŠ
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype02 
dense/Tensordot/ReadVariableOp
dense/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@     2
dense/Tensordot/Reshape/shape
dense/Tensordot/ReshapeReshapeinput_1&dense/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
Ŕ2
dense/Tensordot/ReshapeŽ
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ŕ2
dense/Tensordot/MatMul
dense/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    
      2
dense/Tensordot/shape
dense/TensordotReshape dense/Tensordot/MatMul:product:0dense/Tensordot/shape:output:0*
T0*"
_output_shapes
: 
2
dense/Tensordot
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
: 
2
dense/BiasAdde

dense/ReluReludense/BiasAdd:output:0*
T0*"
_output_shapes
: 
2

dense/Reluw
dropout/IdentityIdentitydense/Relu:activations:0*
T0*"
_output_shapes
: 
2
dropout/Identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙x   2
flatten/Const
flatten/ReshapeReshapedropout/Identity:output:0flatten/Const:output:0*
T0*
_output_shapes

: x2
flatten/ReshapeĽ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:x*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
dense_1/BiasAddp
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*
_output_shapes

: 2
dense_1/Softmaxd
IdentityIdentitydense_1/Softmax:softmax:0*
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
:˙˙˙˙˙˙˙˙˙

!
_user_specified_name	input_1
¤
¤
I__inference_sequential_layer_call_and_return_conditional_losses_152977125

inputs
dense_152977112
dense_152977114
dense_1_152977119
dense_1_152977121
identity˘dense/StatefulPartitionedCall˘dense_1/StatefulPartitionedCall˘dropout/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_152977112dense_152977114*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
: 
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_1529770022
dense/StatefulPartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
: 
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_layer_call_and_return_conditional_losses_1529770302!
dropout/StatefulPartitionedCallí
flatten/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: x* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_layer_call_and_return_conditional_losses_1529770542
flatten/PartitionedCallŠ
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_152977119dense_1_152977121*
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
F__inference_dense_1_layer_call_and_return_conditional_losses_1529770732!
dense_1/StatefulPartitionedCall×
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*2
_input_shapes!
: 
::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙

 
_user_specified_nameinputs
9
˙	
"__inference__traced_save_152977534
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
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
*savev2_false_negatives_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpoints
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
value3B1 B+_temp_e5fa3b2033c643d2906dcd6f4ea9cf9e/part2	
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
ShardedFilename/shardŚ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameĐ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*â
valueŘBŐB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesź
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices˙	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableop+savev2_true_positives_1_read_readvariableop*savev2_false_negatives_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *(
dtypes
2	2
SaveV2ş
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesĄ
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

identity_1Identity_1:output:0*Ś
_input_shapes
: :	::x:: : : : : : : : : :::::	::x::	::x:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	: 

_output_shapes
::$ 

_output_shapes

:x: 
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
::%!

_output_shapes
:	: 

_output_shapes
::$ 

_output_shapes

:x: 

_output_shapes
::%!

_output_shapes
:	: 

_output_shapes
::$ 

_output_shapes

:x: 

_output_shapes
::

_output_shapes
: 
ú
G
+__inference_flatten_layer_call_fn_152977416

inputs
identityť
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: x* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_layer_call_and_return_conditional_losses_1529770542
PartitionedCallc
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes

: x2

Identity"
identityIdentity:output:0*!
_input_shapes
: 
:J F
"
_output_shapes
: 

 
_user_specified_nameinputs

d
+__inference_dropout_layer_call_fn_152977400

inputs
identity˘StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
: 
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_layer_call_and_return_conditional_losses_1529770302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*"
_output_shapes
: 
2

Identity"
identityIdentity:output:0*!
_input_shapes
: 
22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
: 

 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_default
7
input_1,
serving_default_input_1:0 
2
dense_1'
StatefulPartitionedCall:0 tensorflow/serving/predict:
ú
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
*]&call_and_return_all_conditional_losses
^__call__"Ç
_tf_keras_sequential¨{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [32, 10, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 12, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 11, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 10, 128]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [32, 10, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 12, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 11, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy", "Precision", "Recall"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
đ

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*_&call_and_return_all_conditional_losses
`__call__"Ë
_tf_keras_layerą{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 12, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 10, 128]}}
á
	variables
trainable_variables
regularization_losses
	keras_api
*a&call_and_return_all_conditional_losses
b__call__"Ň
_tf_keras_layer¸{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
â
	variables
trainable_variables
regularization_losses
	keras_api
*c&call_and_return_all_conditional_losses
d__call__"Ó
_tf_keras_layerš{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ó

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*e&call_and_return_all_conditional_losses
f__call__"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 11, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 120}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 120]}}
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
Ę
$metrics
	variables
trainable_variables
regularization_losses

%layers
&layer_metrics
'non_trainable_variables
(layer_regularization_losses
^__call__
\_default_save_signature
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
,
gserving_default"
signature_map
:	2dense/kernel
:2
dense/bias
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
)metrics
	variables
trainable_variables
regularization_losses

*layers
+layer_metrics
,non_trainable_variables
-layer_regularization_losses
`__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
.metrics
	variables
trainable_variables
regularization_losses

/layers
0layer_metrics
1non_trainable_variables
2layer_regularization_losses
b__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
3metrics
	variables
trainable_variables
regularization_losses

4layers
5layer_metrics
6non_trainable_variables
7layer_regularization_losses
d__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
 :x2dense_1/kernel
:2dense_1/bias
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
8metrics
	variables
trainable_variables
regularization_losses

9layers
:layer_metrics
;non_trainable_variables
<layer_regularization_losses
f__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
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
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ť
	Atotal
	Bcount
C	variables
D	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
˙
	Etotal
	Fcount
G
_fn_kwargs
H	variables
I	keras_api"¸
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
Ł
J
thresholds
Ktrue_positives
Lfalse_positives
M	variables
N	keras_api"É
_tf_keras_metricŽ{"class_name": "Precision", "name": "precision", "dtype": "float32", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}

O
thresholds
Ptrue_positives
Qfalse_negatives
R	variables
S	keras_api"Ŕ
_tf_keras_metricĽ{"class_name": "Recall", "name": "recall", "dtype": "float32", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
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
$:"	2Adam/dense/kernel/m
:2Adam/dense/bias/m
%:#x2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
$:"	2Adam/dense/kernel/v
:2Adam/dense/bias/v
%:#x2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
ç2ä
$__inference__wrapped_model_152976983ť
˛
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
annotationsŞ *+˘(
&#
input_1˙˙˙˙˙˙˙˙˙

ň2ď
I__inference_sequential_layer_call_and_return_conditional_losses_152977245
I__inference_sequential_layer_call_and_return_conditional_losses_152977328
I__inference_sequential_layer_call_and_return_conditional_losses_152977220
I__inference_sequential_layer_call_and_return_conditional_losses_152977303Ŕ
ˇ˛ł
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
kwonlydefaultsŞ 
annotationsŞ *
 
2
.__inference_sequential_layer_call_fn_152977271
.__inference_sequential_layer_call_fn_152977258
.__inference_sequential_layer_call_fn_152977341
.__inference_sequential_layer_call_fn_152977354Ŕ
ˇ˛ł
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
kwonlydefaultsŞ 
annotationsŞ *
 
î2ë
D__inference_dense_layer_call_and_return_conditional_losses_152977369˘
˛
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
annotationsŞ *
 
Ó2Đ
)__inference_dense_layer_call_fn_152977378˘
˛
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
annotationsŞ *
 
Ę2Ç
F__inference_dropout_layer_call_and_return_conditional_losses_152977395
F__inference_dropout_layer_call_and_return_conditional_losses_152977390´
Ť˛§
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
kwonlydefaultsŞ 
annotationsŞ *
 
2
+__inference_dropout_layer_call_fn_152977405
+__inference_dropout_layer_call_fn_152977400´
Ť˛§
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
kwonlydefaultsŞ 
annotationsŞ *
 
đ2í
F__inference_flatten_layer_call_and_return_conditional_losses_152977411˘
˛
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
annotationsŞ *
 
Ő2Ň
+__inference_flatten_layer_call_fn_152977416˘
˛
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
annotationsŞ *
 
đ2í
F__inference_dense_1_layer_call_and_return_conditional_losses_152977427˘
˛
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
annotationsŞ *
 
Ő2Ň
+__inference_dense_1_layer_call_fn_152977436˘
˛
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
annotationsŞ *
 
6B4
'__inference_signature_wrapper_152977188input_1
$__inference__wrapped_model_152976983g5˘2
+˘(
&#
input_1˙˙˙˙˙˙˙˙˙

Ş "(Ş%
#
dense_1
dense_1 
F__inference_dense_1_layer_call_and_return_conditional_losses_152977427J&˘#
˘

inputs x
Ş "˘

0 
 l
+__inference_dense_1_layer_call_fn_152977436=&˘#
˘

inputs x
Ş " 
D__inference_dense_layer_call_and_return_conditional_losses_152977369S+˘(
!˘

inputs 

Ş " ˘

0 

 s
)__inference_dense_layer_call_fn_152977378F+˘(
!˘

inputs 

Ş " 

F__inference_dropout_layer_call_and_return_conditional_losses_152977390R.˘+
$˘!

inputs 

p
Ş " ˘

0 

 
F__inference_dropout_layer_call_and_return_conditional_losses_152977395R.˘+
$˘!

inputs 

p 
Ş " ˘

0 

 t
+__inference_dropout_layer_call_fn_152977400E.˘+
$˘!

inputs 

p
Ş " 
t
+__inference_dropout_layer_call_fn_152977405E.˘+
$˘!

inputs 

p 
Ş " 

F__inference_flatten_layer_call_and_return_conditional_losses_152977411J*˘'
 ˘

inputs 

Ş "˘

0 x
 l
+__inference_flatten_layer_call_fn_152977416=*˘'
 ˘

inputs 

Ş " xŻ
I__inference_sequential_layer_call_and_return_conditional_losses_152977220b<˘9
2˘/
%"
inputs˙˙˙˙˙˙˙˙˙

p

 
Ş "˘

0 
 Ż
I__inference_sequential_layer_call_and_return_conditional_losses_152977245b<˘9
2˘/
%"
inputs˙˙˙˙˙˙˙˙˙

p 

 
Ş "˘

0 
 °
I__inference_sequential_layer_call_and_return_conditional_losses_152977303c=˘:
3˘0
&#
input_1˙˙˙˙˙˙˙˙˙

p

 
Ş "˘

0 
 °
I__inference_sequential_layer_call_and_return_conditional_losses_152977328c=˘:
3˘0
&#
input_1˙˙˙˙˙˙˙˙˙

p 

 
Ş "˘

0 
 
.__inference_sequential_layer_call_fn_152977258U<˘9
2˘/
%"
inputs˙˙˙˙˙˙˙˙˙

p

 
Ş " 
.__inference_sequential_layer_call_fn_152977271U<˘9
2˘/
%"
inputs˙˙˙˙˙˙˙˙˙

p 

 
Ş " 
.__inference_sequential_layer_call_fn_152977341V=˘:
3˘0
&#
input_1˙˙˙˙˙˙˙˙˙

p

 
Ş " 
.__inference_sequential_layer_call_fn_152977354V=˘:
3˘0
&#
input_1˙˙˙˙˙˙˙˙˙

p 

 
Ş " 
'__inference_signature_wrapper_152977188i7˘4
˘ 
-Ş*
(
input_1
input_1 
"(Ş%
#
dense_1
dense_1 