Û
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
 "serve*2.3.12v2.3.0-54-gfcc4b966f18ÉÃ
z
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_4/kernel
s
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel* 
_output_shapes
:
*
dtype0
q
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes	
:*
dtype0
y
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*
shared_namedense_5/kernel
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes
:	@*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:@*
dtype0
y
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_6/kernel
r
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes
:	*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
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
Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_4/kernel/m

)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/m
x
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*&
shared_nameAdam/dense_5/kernel/m

)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m*
_output_shapes
:	@*
dtype0
~
Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_5/bias/m
w
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/dense_6/kernel/m

)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m*
_output_shapes
:	*
dtype0
~
Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_6/bias/m
w
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
_output_shapes
:*
dtype0

Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_4/kernel/v

)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/v
x
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*&
shared_nameAdam/dense_5/kernel/v

)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v*
_output_shapes
:	@*
dtype0
~
Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_5/bias/v
w
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/dense_6/kernel/v

)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v*
_output_shapes
:	*
dtype0
~
Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_6/bias/v
w
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
1
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Õ0
valueË0BÈ0 BÁ0

layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
 	keras_api
R
!	variables
"trainable_variables
#regularization_losses
$	keras_api
h

%kernel
&bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
¬
+iter

,beta_1

-beta_2
	.decay
/learning_ratemjmkmlmm%mn&movpvqvrvs%vt&vu
*
0
1
2
3
%4
&5
*
0
1
2
3
%4
&5
 
­
0metrics
	variables
	trainable_variables

regularization_losses

1layers
2layer_metrics
3non_trainable_variables
4layer_regularization_losses
 
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
5metrics
	variables
trainable_variables
regularization_losses

6layers
7layer_metrics
8non_trainable_variables
9layer_regularization_losses
 
 
 
­
:metrics
	variables
trainable_variables
regularization_losses

;layers
<layer_metrics
=non_trainable_variables
>layer_regularization_losses
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_5/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
?metrics
	variables
trainable_variables
regularization_losses

@layers
Alayer_metrics
Bnon_trainable_variables
Clayer_regularization_losses
 
 
 
­
Dmetrics
	variables
trainable_variables
regularization_losses

Elayers
Flayer_metrics
Gnon_trainable_variables
Hlayer_regularization_losses
 
 
 
­
Imetrics
!	variables
"trainable_variables
#regularization_losses

Jlayers
Klayer_metrics
Lnon_trainable_variables
Mlayer_regularization_losses
ZX
VARIABLE_VALUEdense_6/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_6/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1

%0
&1
 
­
Nmetrics
'	variables
(trainable_variables
)regularization_losses

Olayers
Player_metrics
Qnon_trainable_variables
Rlayer_regularization_losses
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

S0
T1
U2
V3
*
0
1
2
3
4
5
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
	Wtotal
	Xcount
Y	variables
Z	keras_api
D
	[total
	\count
]
_fn_kwargs
^	variables
_	keras_api
W
`
thresholds
atrue_positives
bfalse_positives
c	variables
d	keras_api
W
e
thresholds
ftrue_positives
gfalse_negatives
h	variables
i	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

W0
X1

Y	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

[0
\1

^	variables
 
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE

a0
b1

c	variables
 
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

f0
g1

h	variables
}{
VARIABLE_VALUEAdam/dense_4/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_5/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_5/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_6/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_4/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_5/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_5/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_6/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
r
serving_default_input_3Placeholder*#
_output_shapes
: 
*
dtype0*
shape: 


StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3dense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_signature_wrapper_384421903
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
´
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp)Adam/dense_5/kernel/m/Read/ReadVariableOp'Adam/dense_5/bias/m/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOp)Adam/dense_5/kernel/v/Read/ReadVariableOp'Adam/dense_5/bias/v/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
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
"__inference__traced_save_384422396
Ã
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1true_positivesfalse_positivestrue_positives_1false_negativesAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/dense_5/kernel/mAdam/dense_5/bias/mAdam/dense_6/kernel/mAdam/dense_6/bias/mAdam/dense_4/kernel/vAdam/dense_4/bias/vAdam/dense_5/kernel/vAdam/dense_5/bias/vAdam/dense_6/kernel/vAdam/dense_6/bias/v*+
Tin$
"2 *
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
%__inference__traced_restore_384422499ëÂ
»
f
H__inference_dropout_2_layer_call_and_return_conditional_losses_384421659

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

I
-__inference_dropout_2_layer_call_fn_384422198

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
H__inference_dropout_2_layer_call_and_return_conditional_losses_3844216592
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
Ô
Ñ
%__inference__traced_restore_384422499
file_prefix#
assignvariableop_dense_4_kernel#
assignvariableop_1_dense_4_bias%
!assignvariableop_2_dense_5_kernel#
assignvariableop_3_dense_5_bias%
!assignvariableop_4_dense_6_kernel#
assignvariableop_5_dense_6_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate
assignvariableop_11_total
assignvariableop_12_count
assignvariableop_13_total_1
assignvariableop_14_count_1&
"assignvariableop_15_true_positives'
#assignvariableop_16_false_positives(
$assignvariableop_17_true_positives_1'
#assignvariableop_18_false_negatives-
)assignvariableop_19_adam_dense_4_kernel_m+
'assignvariableop_20_adam_dense_4_bias_m-
)assignvariableop_21_adam_dense_5_kernel_m+
'assignvariableop_22_adam_dense_5_bias_m-
)assignvariableop_23_adam_dense_6_kernel_m+
'assignvariableop_24_adam_dense_6_bias_m-
)assignvariableop_25_adam_dense_4_kernel_v+
'assignvariableop_26_adam_dense_4_bias_v-
)assignvariableop_27_adam_dense_5_kernel_v+
'assignvariableop_28_adam_dense_5_bias_v-
)assignvariableop_29_adam_dense_6_kernel_v+
'assignvariableop_30_adam_dense_6_bias_v
identity_32¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÎ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesÎ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::*.
dtypes$
"2 	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¤
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_4_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¦
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_5_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¤
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_5_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¦
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_6_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¤
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_6_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6¡
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7£
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8£
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¢
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10®
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¡
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12¡
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13£
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14£
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15ª
AssignVariableOp_15AssignVariableOp"assignvariableop_15_true_positivesIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16«
AssignVariableOp_16AssignVariableOp#assignvariableop_16_false_positivesIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¬
AssignVariableOp_17AssignVariableOp$assignvariableop_17_true_positives_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18«
AssignVariableOp_18AssignVariableOp#assignvariableop_18_false_negativesIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19±
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_dense_4_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¯
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_4_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21±
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_5_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¯
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_5_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23±
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_6_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¯
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_6_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25±
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_dense_4_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¯
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_4_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27±
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_dense_5_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28¯
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_dense_5_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29±
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_6_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30¯
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_6_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_309
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_31û
Identity_32IdentityIdentity_31:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_32"#
identity_32Identity_32:output:0*
_input_shapes
~: :::::::::::::::::::::::::::::::2$
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
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
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

d
H__inference_flatten_2_layer_call_and_return_conditional_losses_384422255

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
Const_
ReshapeReshapeinputsConst:output:0*
T0*
_output_shapes
:	 2	
Reshape\
IdentityIdentityReshape:output:0*
T0*
_output_shapes
:	 2

Identity"
identityIdentity:output:0*!
_input_shapes
: 
@:J F
"
_output_shapes
: 
@
 
_user_specified_nameinputs
Ú.
£
$__inference__wrapped_model_384421607
input_3:
6sequential_2_dense_4_tensordot_readvariableop_resource8
4sequential_2_dense_4_biasadd_readvariableop_resource:
6sequential_2_dense_5_tensordot_readvariableop_resource8
4sequential_2_dense_5_biasadd_readvariableop_resource7
3sequential_2_dense_6_matmul_readvariableop_resource8
4sequential_2_dense_6_biasadd_readvariableop_resource
identity×
-sequential_2/dense_4/Tensordot/ReadVariableOpReadVariableOp6sequential_2_dense_4_tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype02/
-sequential_2/dense_4/Tensordot/ReadVariableOp­
,sequential_2/dense_4/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@     2.
,sequential_2/dense_4/Tensordot/Reshape/shapeÆ
&sequential_2/dense_4/Tensordot/ReshapeReshapeinput_35sequential_2/dense_4/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
À2(
&sequential_2/dense_4/Tensordot/Reshapeë
%sequential_2/dense_4/Tensordot/MatMulMatMul/sequential_2/dense_4/Tensordot/Reshape:output:05sequential_2/dense_4/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
À2'
%sequential_2/dense_4/Tensordot/MatMul¡
$sequential_2/dense_4/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    
      2&
$sequential_2/dense_4/Tensordot/shapeÙ
sequential_2/dense_4/TensordotReshape/sequential_2/dense_4/Tensordot/MatMul:product:0-sequential_2/dense_4/Tensordot/shape:output:0*
T0*#
_output_shapes
: 
2 
sequential_2/dense_4/TensordotÌ
+sequential_2/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_2/dense_4/BiasAdd/ReadVariableOpÓ
sequential_2/dense_4/BiasAddBiasAdd'sequential_2/dense_4/Tensordot:output:03sequential_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
: 
2
sequential_2/dense_4/BiasAdd
sequential_2/dense_4/ReluRelu%sequential_2/dense_4/BiasAdd:output:0*
T0*#
_output_shapes
: 
2
sequential_2/dense_4/Relu¥
sequential_2/dropout_2/IdentityIdentity'sequential_2/dense_4/Relu:activations:0*
T0*#
_output_shapes
: 
2!
sequential_2/dropout_2/IdentityÖ
-sequential_2/dense_5/Tensordot/ReadVariableOpReadVariableOp6sequential_2_dense_5_tensordot_readvariableop_resource*
_output_shapes
:	@*
dtype02/
-sequential_2/dense_5/Tensordot/ReadVariableOp­
,sequential_2/dense_5/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@     2.
,sequential_2/dense_5/Tensordot/Reshape/shapeç
&sequential_2/dense_5/Tensordot/ReshapeReshape(sequential_2/dropout_2/Identity:output:05sequential_2/dense_5/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
À2(
&sequential_2/dense_5/Tensordot/Reshapeê
%sequential_2/dense_5/Tensordot/MatMulMatMul/sequential_2/dense_5/Tensordot/Reshape:output:05sequential_2/dense_5/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	À@2'
%sequential_2/dense_5/Tensordot/MatMul¡
$sequential_2/dense_5/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    
   @   2&
$sequential_2/dense_5/Tensordot/shapeØ
sequential_2/dense_5/TensordotReshape/sequential_2/dense_5/Tensordot/MatMul:product:0-sequential_2/dense_5/Tensordot/shape:output:0*
T0*"
_output_shapes
: 
@2 
sequential_2/dense_5/TensordotË
+sequential_2/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+sequential_2/dense_5/BiasAdd/ReadVariableOpÒ
sequential_2/dense_5/BiasAddBiasAdd'sequential_2/dense_5/Tensordot:output:03sequential_2/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
: 
@2
sequential_2/dense_5/BiasAdd
sequential_2/dense_5/ReluRelu%sequential_2/dense_5/BiasAdd:output:0*
T0*"
_output_shapes
: 
@2
sequential_2/dense_5/Relu¤
sequential_2/dropout_3/IdentityIdentity'sequential_2/dense_5/Relu:activations:0*
T0*"
_output_shapes
: 
@2!
sequential_2/dropout_3/Identity
sequential_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
sequential_2/flatten_2/ConstÆ
sequential_2/flatten_2/ReshapeReshape(sequential_2/dropout_3/Identity:output:0%sequential_2/flatten_2/Const:output:0*
T0*
_output_shapes
:	 2 
sequential_2/flatten_2/ReshapeÍ
*sequential_2/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_6_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02,
*sequential_2/dense_6/MatMul/ReadVariableOpÊ
sequential_2/dense_6/MatMulMatMul'sequential_2/flatten_2/Reshape:output:02sequential_2/dense_6/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
sequential_2/dense_6/MatMulË
+sequential_2/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_2/dense_6/BiasAdd/ReadVariableOpÌ
sequential_2/dense_6/BiasAddBiasAdd%sequential_2/dense_6/MatMul:product:03sequential_2/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
sequential_2/dense_6/BiasAdd
sequential_2/dense_6/SoftmaxSoftmax%sequential_2/dense_6/BiasAdd:output:0*
T0*
_output_shapes

: 2
sequential_2/dense_6/Softmaxq
IdentityIdentity&sequential_2/dense_6/Softmax:softmax:0*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*:
_input_shapes)
': 
:::::::U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_3
8
û
K__inference_sequential_2_layer_call_and_return_conditional_losses_384422076

inputs-
)dense_4_tensordot_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource-
)dense_5_tensordot_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
identity°
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 dense_4/Tensordot/ReadVariableOp
dense_4/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@     2!
dense_4/Tensordot/Reshape/shape
dense_4/Tensordot/ReshapeReshapeinputs(dense_4/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
À2
dense_4/Tensordot/Reshape·
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
À2
dense_4/Tensordot/MatMul
dense_4/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    
      2
dense_4/Tensordot/shape¥
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0 dense_4/Tensordot/shape:output:0*
T0*#
_output_shapes
: 
2
dense_4/Tensordot¥
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
: 
2
dense_4/BiasAddl
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*#
_output_shapes
: 
2
dense_4/Reluw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_2/dropout/Const¡
dropout_2/dropout/MulMuldense_4/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*#
_output_shapes
: 
2
dropout_2/dropout/Mul
dropout_2/dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"    
      2
dropout_2/dropout/ShapeÎ
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*#
_output_shapes
: 
*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_2/dropout/GreaterEqual/yâ
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
: 
2 
dropout_2/dropout/GreaterEqual
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
: 
2
dropout_2/dropout/Cast
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*#
_output_shapes
: 
2
dropout_2/dropout/Mul_1¯
 dense_5/Tensordot/ReadVariableOpReadVariableOp)dense_5_tensordot_readvariableop_resource*
_output_shapes
:	@*
dtype02"
 dense_5/Tensordot/ReadVariableOp
dense_5/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@     2!
dense_5/Tensordot/Reshape/shape³
dense_5/Tensordot/ReshapeReshapedropout_2/dropout/Mul_1:z:0(dense_5/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
À2
dense_5/Tensordot/Reshape¶
dense_5/Tensordot/MatMulMatMul"dense_5/Tensordot/Reshape:output:0(dense_5/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	À@2
dense_5/Tensordot/MatMul
dense_5/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    
   @   2
dense_5/Tensordot/shape¤
dense_5/TensordotReshape"dense_5/Tensordot/MatMul:product:0 dense_5/Tensordot/shape:output:0*
T0*"
_output_shapes
: 
@2
dense_5/Tensordot¤
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_5/BiasAdd/ReadVariableOp
dense_5/BiasAddBiasAdddense_5/Tensordot:output:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
: 
@2
dense_5/BiasAddk
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*"
_output_shapes
: 
@2
dense_5/Reluw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_3/dropout/Const 
dropout_3/dropout/MulMuldense_5/Relu:activations:0 dropout_3/dropout/Const:output:0*
T0*"
_output_shapes
: 
@2
dropout_3/dropout/Mul
dropout_3/dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"    
   @   2
dropout_3/dropout/ShapeÍ
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*"
_output_shapes
: 
@*
dtype020
.dropout_3/dropout/random_uniform/RandomUniform
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_3/dropout/GreaterEqual/yá
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*"
_output_shapes
: 
@2 
dropout_3/dropout/GreaterEqual
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*"
_output_shapes
: 
@2
dropout_3/dropout/Cast
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*"
_output_shapes
: 
@2
dropout_3/dropout/Mul_1s
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
flatten_2/Const
flatten_2/ReshapeReshapedropout_3/dropout/Mul_1:z:0flatten_2/Const:output:0*
T0*
_output_shapes
:	 2
flatten_2/Reshape¦
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_6/MatMul/ReadVariableOp
dense_6/MatMulMatMulflatten_2/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
dense_6/MatMul¤
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
dense_6/BiasAddp
dense_6/SoftmaxSoftmaxdense_6/BiasAdd:output:0*
T0*
_output_shapes

: 2
dense_6/Softmaxd
IdentityIdentitydense_6/Softmax:softmax:0*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*:
_input_shapes)
': 
:::::::T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs

d
H__inference_flatten_2_layer_call_and_return_conditional_losses_384421739

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
Const_
ReshapeReshapeinputsConst:output:0*
T0*
_output_shapes
:	 2	
Reshape\
IdentityIdentityReshape:output:0*
T0*
_output_shapes
:	 2

Identity"
identityIdentity:output:0*!
_input_shapes
: 
@:J F
"
_output_shapes
: 
@
 
_user_specified_nameinputs
Ñ

+__inference_dense_4_layer_call_fn_384422171

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
F__inference_dense_4_layer_call_and_return_conditional_losses_3844216262
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
ã
Á
0__inference_sequential_2_layer_call_fn_384422147

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_2_layer_call_and_return_conditional_losses_3844218612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿ
::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Â%
ü
K__inference_sequential_2_layer_call_and_return_conditional_losses_384421991
input_3-
)dense_4_tensordot_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource-
)dense_5_tensordot_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
identity°
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 dense_4/Tensordot/ReadVariableOp
dense_4/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@     2!
dense_4/Tensordot/Reshape/shape
dense_4/Tensordot/ReshapeReshapeinput_3(dense_4/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
À2
dense_4/Tensordot/Reshape·
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
À2
dense_4/Tensordot/MatMul
dense_4/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    
      2
dense_4/Tensordot/shape¥
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0 dense_4/Tensordot/shape:output:0*
T0*#
_output_shapes
: 
2
dense_4/Tensordot¥
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
: 
2
dense_4/BiasAddl
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*#
_output_shapes
: 
2
dense_4/Relu~
dropout_2/IdentityIdentitydense_4/Relu:activations:0*
T0*#
_output_shapes
: 
2
dropout_2/Identity¯
 dense_5/Tensordot/ReadVariableOpReadVariableOp)dense_5_tensordot_readvariableop_resource*
_output_shapes
:	@*
dtype02"
 dense_5/Tensordot/ReadVariableOp
dense_5/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@     2!
dense_5/Tensordot/Reshape/shape³
dense_5/Tensordot/ReshapeReshapedropout_2/Identity:output:0(dense_5/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
À2
dense_5/Tensordot/Reshape¶
dense_5/Tensordot/MatMulMatMul"dense_5/Tensordot/Reshape:output:0(dense_5/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	À@2
dense_5/Tensordot/MatMul
dense_5/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    
   @   2
dense_5/Tensordot/shape¤
dense_5/TensordotReshape"dense_5/Tensordot/MatMul:product:0 dense_5/Tensordot/shape:output:0*
T0*"
_output_shapes
: 
@2
dense_5/Tensordot¤
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_5/BiasAdd/ReadVariableOp
dense_5/BiasAddBiasAdddense_5/Tensordot:output:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
: 
@2
dense_5/BiasAddk
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*"
_output_shapes
: 
@2
dense_5/Relu}
dropout_3/IdentityIdentitydense_5/Relu:activations:0*
T0*"
_output_shapes
: 
@2
dropout_3/Identitys
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
flatten_2/Const
flatten_2/ReshapeReshapedropout_3/Identity:output:0flatten_2/Const:output:0*
T0*
_output_shapes
:	 2
flatten_2/Reshape¦
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_6/MatMul/ReadVariableOp
dense_6/MatMulMatMulflatten_2/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
dense_6/MatMul¤
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
dense_6/BiasAddp
dense_6/SoftmaxSoftmaxdense_6/BiasAdd:output:0*
T0*
_output_shapes

: 2
dense_6/Softmaxd
IdentityIdentitydense_6/Softmax:softmax:0*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*:
_input_shapes)
': 
:::::::U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_3
¤
¹
'__inference_signature_wrapper_384421903
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__wrapped_model_3844216072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*:
_input_shapes)
': 
::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
: 

!
_user_specified_name	input_3
ÿ
®
F__inference_dense_6_layer_call_and_return_conditional_losses_384422271

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
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
:	 :::G C

_output_shapes
:	 
 
_user_specified_nameinputs
Ü
±
F__inference_dense_4_layer_call_and_return_conditional_losses_384422162

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
ÿ
®
F__inference_dense_6_layer_call_and_return_conditional_losses_384421758

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
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
:	 :::G C

_output_shapes
:	 
 
_user_specified_nameinputs

g
H__inference_dropout_2_layer_call_and_return_conditional_losses_384421654

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
·
f
H__inference_dropout_3_layer_call_and_return_conditional_losses_384421720

inputs

identity_1U
IdentityIdentityinputs*
T0*"
_output_shapes
: 
@2

Identityd

Identity_1IdentityIdentity:output:0*
T0*"
_output_shapes
: 
@2

Identity_1"!

identity_1Identity_1:output:0*!
_input_shapes
: 
@:J F
"
_output_shapes
: 
@
 
_user_specified_nameinputs

I
-__inference_dropout_3_layer_call_fn_384422249

inputs
identityÁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
: 
@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_3_layer_call_and_return_conditional_losses_3844217202
PartitionedCallg
IdentityIdentityPartitionedCall:output:0*
T0*"
_output_shapes
: 
@2

Identity"
identityIdentity:output:0*!
_input_shapes
: 
@:J F
"
_output_shapes
: 
@
 
_user_specified_nameinputs

Ú
K__inference_sequential_2_layer_call_and_return_conditional_losses_384421861

inputs
dense_4_384421842
dense_4_384421844
dense_5_384421848
dense_5_384421850
dense_6_384421855
dense_6_384421857
identity¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall
dense_4/StatefulPartitionedCallStatefulPartitionedCallinputsdense_4_384421842dense_4_384421844*
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
F__inference_dense_4_layer_call_and_return_conditional_losses_3844216262!
dense_4/StatefulPartitionedCallø
dropout_2/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
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
H__inference_dropout_2_layer_call_and_return_conditional_losses_3844216592
dropout_2/PartitionedCall¯
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_5_384421848dense_5_384421850*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
: 
@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_5_layer_call_and_return_conditional_losses_3844216872!
dense_5/StatefulPartitionedCall÷
dropout_3/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
: 
@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_3_layer_call_and_return_conditional_losses_3844217202
dropout_3/PartitionedCallî
flatten_2/PartitionedCallPartitionedCall"dropout_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_flatten_2_layer_call_and_return_conditional_losses_3844217392
flatten_2/PartitionedCall«
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_6_384421855dense_6_384421857*
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
F__inference_dense_6_layer_call_and_return_conditional_losses_3844217582!
dense_6/StatefulPartitionedCallÙ
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*:
_input_shapes)
': 
::::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ï

+__inference_dense_5_layer_call_fn_384422222

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
: 
@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_5_layer_call_and_return_conditional_losses_3844216872
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*"
_output_shapes
: 
@2

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
û

g
H__inference_dropout_3_layer_call_and_return_conditional_losses_384421715

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
@2
dropout/Muls
dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"    
   @   2
dropout/Shape¯
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*"
_output_shapes
: 
@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y¹
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*"
_output_shapes
: 
@2
dropout/GreaterEqualz
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*"
_output_shapes
: 
@2
dropout/Castu
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*"
_output_shapes
: 
@2
dropout/Mul_1`
IdentityIdentitydropout/Mul_1:z:0*
T0*"
_output_shapes
: 
@2

Identity"
identityIdentity:output:0*!
_input_shapes
: 
@:J F
"
_output_shapes
: 
@
 
_user_specified_nameinputs
û

g
H__inference_dropout_3_layer_call_and_return_conditional_losses_384422234

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
@2
dropout/Muls
dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"    
   @   2
dropout/Shape¯
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*"
_output_shapes
: 
@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y¹
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*"
_output_shapes
: 
@2
dropout/GreaterEqualz
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*"
_output_shapes
: 
@2
dropout/Castu
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*"
_output_shapes
: 
@2
dropout/Mul_1`
IdentityIdentitydropout/Mul_1:z:0*
T0*"
_output_shapes
: 
@2

Identity"
identityIdentity:output:0*!
_input_shapes
: 
@:J F
"
_output_shapes
: 
@
 
_user_specified_nameinputs
Ü
±
F__inference_dense_4_layer_call_and_return_conditional_losses_384421626

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
ã
Á
0__inference_sequential_2_layer_call_fn_384422130

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_2_layer_call_and_return_conditional_losses_3844218222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿ
::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
8
ü
K__inference_sequential_2_layer_call_and_return_conditional_losses_384421954
input_3-
)dense_4_tensordot_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource-
)dense_5_tensordot_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
identity°
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 dense_4/Tensordot/ReadVariableOp
dense_4/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@     2!
dense_4/Tensordot/Reshape/shape
dense_4/Tensordot/ReshapeReshapeinput_3(dense_4/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
À2
dense_4/Tensordot/Reshape·
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
À2
dense_4/Tensordot/MatMul
dense_4/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    
      2
dense_4/Tensordot/shape¥
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0 dense_4/Tensordot/shape:output:0*
T0*#
_output_shapes
: 
2
dense_4/Tensordot¥
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
: 
2
dense_4/BiasAddl
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*#
_output_shapes
: 
2
dense_4/Reluw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_2/dropout/Const¡
dropout_2/dropout/MulMuldense_4/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*#
_output_shapes
: 
2
dropout_2/dropout/Mul
dropout_2/dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"    
      2
dropout_2/dropout/ShapeÎ
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*#
_output_shapes
: 
*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_2/dropout/GreaterEqual/yâ
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
: 
2 
dropout_2/dropout/GreaterEqual
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*#
_output_shapes
: 
2
dropout_2/dropout/Cast
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*#
_output_shapes
: 
2
dropout_2/dropout/Mul_1¯
 dense_5/Tensordot/ReadVariableOpReadVariableOp)dense_5_tensordot_readvariableop_resource*
_output_shapes
:	@*
dtype02"
 dense_5/Tensordot/ReadVariableOp
dense_5/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@     2!
dense_5/Tensordot/Reshape/shape³
dense_5/Tensordot/ReshapeReshapedropout_2/dropout/Mul_1:z:0(dense_5/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
À2
dense_5/Tensordot/Reshape¶
dense_5/Tensordot/MatMulMatMul"dense_5/Tensordot/Reshape:output:0(dense_5/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	À@2
dense_5/Tensordot/MatMul
dense_5/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    
   @   2
dense_5/Tensordot/shape¤
dense_5/TensordotReshape"dense_5/Tensordot/MatMul:product:0 dense_5/Tensordot/shape:output:0*
T0*"
_output_shapes
: 
@2
dense_5/Tensordot¤
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_5/BiasAdd/ReadVariableOp
dense_5/BiasAddBiasAdddense_5/Tensordot:output:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
: 
@2
dense_5/BiasAddk
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*"
_output_shapes
: 
@2
dense_5/Reluw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_3/dropout/Const 
dropout_3/dropout/MulMuldense_5/Relu:activations:0 dropout_3/dropout/Const:output:0*
T0*"
_output_shapes
: 
@2
dropout_3/dropout/Mul
dropout_3/dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"    
   @   2
dropout_3/dropout/ShapeÍ
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*"
_output_shapes
: 
@*
dtype020
.dropout_3/dropout/random_uniform/RandomUniform
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_3/dropout/GreaterEqual/yá
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*"
_output_shapes
: 
@2 
dropout_3/dropout/GreaterEqual
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*"
_output_shapes
: 
@2
dropout_3/dropout/Cast
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*"
_output_shapes
: 
@2
dropout_3/dropout/Mul_1s
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
flatten_2/Const
flatten_2/ReshapeReshapedropout_3/dropout/Mul_1:z:0flatten_2/Const:output:0*
T0*
_output_shapes
:	 2
flatten_2/Reshape¦
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_6/MatMul/ReadVariableOp
dense_6/MatMulMatMulflatten_2/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
dense_6/MatMul¤
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
dense_6/BiasAddp
dense_6/SoftmaxSoftmaxdense_6/BiasAdd:output:0*
T0*
_output_shapes

: 2
dense_6/Softmaxd
IdentityIdentitydense_6/Softmax:softmax:0*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*:
_input_shapes)
': 
:::::::U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_3

g
H__inference_dropout_2_layer_call_and_return_conditional_losses_384422183

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

f
-__inference_dropout_3_layer_call_fn_384422244

inputs
identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
: 
@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_3_layer_call_and_return_conditional_losses_3844217152
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*"
_output_shapes
: 
@2

Identity"
identityIdentity:output:0*!
_input_shapes
: 
@22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
: 
@
 
_user_specified_nameinputs
¿%
û
K__inference_sequential_2_layer_call_and_return_conditional_losses_384422113

inputs-
)dense_4_tensordot_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource-
)dense_5_tensordot_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
identity°
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 dense_4/Tensordot/ReadVariableOp
dense_4/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@     2!
dense_4/Tensordot/Reshape/shape
dense_4/Tensordot/ReshapeReshapeinputs(dense_4/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
À2
dense_4/Tensordot/Reshape·
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0* 
_output_shapes
:
À2
dense_4/Tensordot/MatMul
dense_4/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    
      2
dense_4/Tensordot/shape¥
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0 dense_4/Tensordot/shape:output:0*
T0*#
_output_shapes
: 
2
dense_4/Tensordot¥
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
: 
2
dense_4/BiasAddl
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*#
_output_shapes
: 
2
dense_4/Relu~
dropout_2/IdentityIdentitydense_4/Relu:activations:0*
T0*#
_output_shapes
: 
2
dropout_2/Identity¯
 dense_5/Tensordot/ReadVariableOpReadVariableOp)dense_5_tensordot_readvariableop_resource*
_output_shapes
:	@*
dtype02"
 dense_5/Tensordot/ReadVariableOp
dense_5/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@     2!
dense_5/Tensordot/Reshape/shape³
dense_5/Tensordot/ReshapeReshapedropout_2/Identity:output:0(dense_5/Tensordot/Reshape/shape:output:0*
T0* 
_output_shapes
:
À2
dense_5/Tensordot/Reshape¶
dense_5/Tensordot/MatMulMatMul"dense_5/Tensordot/Reshape:output:0(dense_5/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	À@2
dense_5/Tensordot/MatMul
dense_5/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    
   @   2
dense_5/Tensordot/shape¤
dense_5/TensordotReshape"dense_5/Tensordot/MatMul:product:0 dense_5/Tensordot/shape:output:0*
T0*"
_output_shapes
: 
@2
dense_5/Tensordot¤
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_5/BiasAdd/ReadVariableOp
dense_5/BiasAddBiasAdddense_5/Tensordot:output:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
: 
@2
dense_5/BiasAddk
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*"
_output_shapes
: 
@2
dense_5/Relu}
dropout_3/IdentityIdentitydense_5/Relu:activations:0*
T0*"
_output_shapes
: 
@2
dropout_3/Identitys
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
flatten_2/Const
flatten_2/ReshapeReshapedropout_3/Identity:output:0flatten_2/Const:output:0*
T0*
_output_shapes
:	 2
flatten_2/Reshape¦
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_6/MatMul/ReadVariableOp
dense_6/MatMulMatMulflatten_2/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
dense_6/MatMul¤
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
dense_6/BiasAddp
dense_6/SoftmaxSoftmaxdense_6/BiasAdd:output:0*
T0*
_output_shapes

: 2
dense_6/Softmaxd
IdentityIdentitydense_6/Softmax:softmax:0*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*:
_input_shapes)
': 
:::::::T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
æ
Â
0__inference_sequential_2_layer_call_fn_384422025
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_2_layer_call_and_return_conditional_losses_3844218612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿ
::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_3
·
f
H__inference_dropout_3_layer_call_and_return_conditional_losses_384422239

inputs

identity_1U
IdentityIdentityinputs*
T0*"
_output_shapes
: 
@2

Identityd

Identity_1IdentityIdentity:output:0*
T0*"
_output_shapes
: 
@2

Identity_1"!

identity_1Identity_1:output:0*!
_input_shapes
: 
@:J F
"
_output_shapes
: 
@
 
_user_specified_nameinputs
ÙC
»
"__inference__traced_save_384422396
file_prefix-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop(
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
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop4
0savev2_adam_dense_5_kernel_m_read_readvariableop2
.savev2_adam_dense_5_bias_m_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop4
0savev2_adam_dense_5_kernel_v_read_readvariableop2
.savev2_adam_dense_5_bias_v_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableop
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
value3B1 B+_temp_3d35e83c3008411996ea352588a0fa41/part2	
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
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesÈ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices©
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableop+savev2_true_positives_1_read_readvariableop*savev2_false_negatives_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	2
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

identity_1Identity_1:output:0*â
_input_shapesÐ
Í: :
::	@:@:	:: : : : : : : : : :::::
::	@:@:	::
::	@:@:	:: 2(
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
:	@: 

_output_shapes
:@:%!

_output_shapes
:	: 

_output_shapes
::
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
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	@: 

_output_shapes
:@:%!

_output_shapes
:	: 

_output_shapes
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	@: 

_output_shapes
:@:%!

_output_shapes
:	: 

_output_shapes
:: 

_output_shapes
: 
æ
Â
0__inference_sequential_2_layer_call_fn_384422008
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_2_layer_call_and_return_conditional_losses_3844218222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿ
::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_3

¢
K__inference_sequential_2_layer_call_and_return_conditional_losses_384421822

inputs
dense_4_384421803
dense_4_384421805
dense_5_384421809
dense_5_384421811
dense_6_384421816
dense_6_384421818
identity¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢!dropout_2/StatefulPartitionedCall¢!dropout_3/StatefulPartitionedCall
dense_4/StatefulPartitionedCallStatefulPartitionedCallinputsdense_4_384421803dense_4_384421805*
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
F__inference_dense_4_layer_call_and_return_conditional_losses_3844216262!
dense_4/StatefulPartitionedCall
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
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
H__inference_dropout_2_layer_call_and_return_conditional_losses_3844216542#
!dropout_2/StatefulPartitionedCall·
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_5_384421809dense_5_384421811*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
: 
@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_5_layer_call_and_return_conditional_losses_3844216872!
dense_5/StatefulPartitionedCall³
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
: 
@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_3_layer_call_and_return_conditional_losses_3844217152#
!dropout_3/StatefulPartitionedCallö
flatten_2/PartitionedCallPartitionedCall*dropout_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_flatten_2_layer_call_and_return_conditional_losses_3844217392
flatten_2/PartitionedCall«
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_6_384421816dense_6_384421818*
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
F__inference_dense_6_layer_call_and_return_conditional_losses_3844217582!
dense_6/StatefulPartitionedCall¡
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*:
_input_shapes)
': 
::::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs

f
-__inference_dropout_2_layer_call_fn_384422193

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
H__inference_dropout_2_layer_call_and_return_conditional_losses_3844216542
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
¿

+__inference_dense_6_layer_call_fn_384422280

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
F__inference_dense_6_layer_call_and_return_conditional_losses_3844217582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*&
_input_shapes
:	 ::22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	 
 
_user_specified_nameinputs
»
f
H__inference_dropout_2_layer_call_and_return_conditional_losses_384422188

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
Õ
±
F__inference_dense_5_layer_call_and_return_conditional_losses_384421687

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	@*
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
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	À@2
Tensordot/MatMulw
Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    
   @   2
Tensordot/shape
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*"
_output_shapes
: 
@2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp~
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
: 
@2	
BiasAddS
ReluReluBiasAdd:output:0*
T0*"
_output_shapes
: 
@2
Relua
IdentityIdentityRelu:activations:0*
T0*"
_output_shapes
: 
@2

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
Õ
±
F__inference_dense_5_layer_call_and_return_conditional_losses_384422213

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	@*
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
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	À@2
Tensordot/MatMulw
Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    
   @   2
Tensordot/shape
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*"
_output_shapes
: 
@2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp~
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
: 
@2	
BiasAddS
ReluReluBiasAdd:output:0*
T0*"
_output_shapes
: 
@2
Relua
IdentityIdentityRelu:activations:0*
T0*"
_output_shapes
: 
@2

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

I
-__inference_flatten_2_layer_call_fn_384422260

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
:	 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_flatten_2_layer_call_and_return_conditional_losses_3844217392
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	 2

Identity"
identityIdentity:output:0*!
_input_shapes
: 
@:J F
"
_output_shapes
: 
@
 
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
input_3,
serving_default_input_3:0 
2
dense_6'
StatefulPartitionedCall:0 tensorflow/serving/predict:ëÂ
¸(
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api

signatures
v_default_save_signature
*w&call_and_return_all_conditional_losses
x__call__"Ñ%
_tf_keras_sequential²%{"class_name": "Sequential", "name": "sequential_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [32, 10, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 11, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 10, 128]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [32, 10, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 11, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy", "Precision", "Recall"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
õ

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*y&call_and_return_all_conditional_losses
z__call__"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 10, 128]}}
å
	variables
trainable_variables
regularization_losses
	keras_api
*{&call_and_return_all_conditional_losses
|__call__"Ö
_tf_keras_layer¼{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
ô

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*}&call_and_return_all_conditional_losses
~__call__"Ï
_tf_keras_layerµ{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 10, 128]}}
æ
	variables
trainable_variables
regularization_losses
 	keras_api
*&call_and_return_all_conditional_losses
__call__"Ö
_tf_keras_layer¼{"class_name": "Dropout", "name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
è
!	variables
"trainable_variables
#regularization_losses
$	keras_api
+&call_and_return_all_conditional_losses
__call__"×
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
õ

%kernel
&bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+&call_and_return_all_conditional_losses
__call__"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 11, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 640}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 640]}}
¿
+iter

,beta_1

-beta_2
	.decay
/learning_ratemjmkmlmm%mn&movpvqvrvs%vt&vu"
	optimizer
J
0
1
2
3
%4
&5"
trackable_list_wrapper
J
0
1
2
3
%4
&5"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
0metrics
	variables
	trainable_variables

regularization_losses

1layers
2layer_metrics
3non_trainable_variables
4layer_regularization_losses
x__call__
v_default_save_signature
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
": 
2dense_4/kernel
:2dense_4/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
5metrics
	variables
trainable_variables
regularization_losses

6layers
7layer_metrics
8non_trainable_variables
9layer_regularization_losses
z__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
:metrics
	variables
trainable_variables
regularization_losses

;layers
<layer_metrics
=non_trainable_variables
>layer_regularization_losses
|__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
!:	@2dense_5/kernel
:@2dense_5/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
?metrics
	variables
trainable_variables
regularization_losses

@layers
Alayer_metrics
Bnon_trainable_variables
Clayer_regularization_losses
~__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
®
Dmetrics
	variables
trainable_variables
regularization_losses

Elayers
Flayer_metrics
Gnon_trainable_variables
Hlayer_regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
Imetrics
!	variables
"trainable_variables
#regularization_losses

Jlayers
Klayer_metrics
Lnon_trainable_variables
Mlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!:	2dense_6/kernel
:2dense_6/bias
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Nmetrics
'	variables
(trainable_variables
)regularization_losses

Olayers
Player_metrics
Qnon_trainable_variables
Rlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
<
S0
T1
U2
V3"
trackable_list_wrapper
J
0
1
2
3
4
5"
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
»
	Wtotal
	Xcount
Y	variables
Z	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
ÿ
	[total
	\count
]
_fn_kwargs
^	variables
_	keras_api"¸
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
£
`
thresholds
atrue_positives
bfalse_positives
c	variables
d	keras_api"É
_tf_keras_metric®{"class_name": "Precision", "name": "precision", "dtype": "float32", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}

e
thresholds
ftrue_positives
gfalse_negatives
h	variables
i	keras_api"À
_tf_keras_metric¥{"class_name": "Recall", "name": "recall", "dtype": "float32", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
:  (2total
:  (2count
.
W0
X1"
trackable_list_wrapper
-
Y	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
[0
\1"
trackable_list_wrapper
-
^	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
.
a0
b1"
trackable_list_wrapper
-
c	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
.
f0
g1"
trackable_list_wrapper
-
h	variables"
_generic_user_object
':%
2Adam/dense_4/kernel/m
 :2Adam/dense_4/bias/m
&:$	@2Adam/dense_5/kernel/m
:@2Adam/dense_5/bias/m
&:$	2Adam/dense_6/kernel/m
:2Adam/dense_6/bias/m
':%
2Adam/dense_4/kernel/v
 :2Adam/dense_4/bias/v
&:$	@2Adam/dense_5/kernel/v
:@2Adam/dense_5/bias/v
&:$	2Adam/dense_6/kernel/v
:2Adam/dense_6/bias/v
ç2ä
$__inference__wrapped_model_384421607»
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
input_3ÿÿÿÿÿÿÿÿÿ

ú2÷
K__inference_sequential_2_layer_call_and_return_conditional_losses_384421991
K__inference_sequential_2_layer_call_and_return_conditional_losses_384422076
K__inference_sequential_2_layer_call_and_return_conditional_losses_384421954
K__inference_sequential_2_layer_call_and_return_conditional_losses_384422113À
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
2
0__inference_sequential_2_layer_call_fn_384422130
0__inference_sequential_2_layer_call_fn_384422025
0__inference_sequential_2_layer_call_fn_384422008
0__inference_sequential_2_layer_call_fn_384422147À
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
ð2í
F__inference_dense_4_layer_call_and_return_conditional_losses_384422162¢
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
+__inference_dense_4_layer_call_fn_384422171¢
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
Î2Ë
H__inference_dropout_2_layer_call_and_return_conditional_losses_384422188
H__inference_dropout_2_layer_call_and_return_conditional_losses_384422183´
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
2
-__inference_dropout_2_layer_call_fn_384422198
-__inference_dropout_2_layer_call_fn_384422193´
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
ð2í
F__inference_dense_5_layer_call_and_return_conditional_losses_384422213¢
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
+__inference_dense_5_layer_call_fn_384422222¢
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
Î2Ë
H__inference_dropout_3_layer_call_and_return_conditional_losses_384422234
H__inference_dropout_3_layer_call_and_return_conditional_losses_384422239´
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
2
-__inference_dropout_3_layer_call_fn_384422244
-__inference_dropout_3_layer_call_fn_384422249´
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
ò2ï
H__inference_flatten_2_layer_call_and_return_conditional_losses_384422255¢
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
×2Ô
-__inference_flatten_2_layer_call_fn_384422260¢
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
F__inference_dense_6_layer_call_and_return_conditional_losses_384422271¢
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
+__inference_dense_6_layer_call_fn_384422280¢
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
'__inference_signature_wrapper_384421903input_3
$__inference__wrapped_model_384421607i%&5¢2
+¢(
&#
input_3ÿÿÿÿÿÿÿÿÿ

ª "(ª%
#
dense_6
dense_6 
F__inference_dense_4_layer_call_and_return_conditional_losses_384422162T+¢(
!¢

inputs 

ª "!¢

0 

 v
+__inference_dense_4_layer_call_fn_384422171G+¢(
!¢

inputs 

ª " 

F__inference_dense_5_layer_call_and_return_conditional_losses_384422213S+¢(
!¢

inputs 

ª " ¢

0 
@
 u
+__inference_dense_5_layer_call_fn_384422222F+¢(
!¢

inputs 

ª " 
@
F__inference_dense_6_layer_call_and_return_conditional_losses_384422271K%&'¢$
¢

inputs	 
ª "¢

0 
 m
+__inference_dense_6_layer_call_fn_384422280>%&'¢$
¢

inputs	 
ª "  
H__inference_dropout_2_layer_call_and_return_conditional_losses_384422183T/¢,
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
H__inference_dropout_2_layer_call_and_return_conditional_losses_384422188T/¢,
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
-__inference_dropout_2_layer_call_fn_384422193G/¢,
%¢"

inputs 

p
ª " 
x
-__inference_dropout_2_layer_call_fn_384422198G/¢,
%¢"

inputs 

p 
ª " 

H__inference_dropout_3_layer_call_and_return_conditional_losses_384422234R.¢+
$¢!

inputs 
@
p
ª " ¢

0 
@
 
H__inference_dropout_3_layer_call_and_return_conditional_losses_384422239R.¢+
$¢!

inputs 
@
p 
ª " ¢

0 
@
 v
-__inference_dropout_3_layer_call_fn_384422244E.¢+
$¢!

inputs 
@
p
ª " 
@v
-__inference_dropout_3_layer_call_fn_384422249E.¢+
$¢!

inputs 
@
p 
ª " 
@
H__inference_flatten_2_layer_call_and_return_conditional_losses_384422255K*¢'
 ¢

inputs 
@
ª "¢

0	 
 o
-__inference_flatten_2_layer_call_fn_384422260>*¢'
 ¢

inputs 
@
ª "	 ´
K__inference_sequential_2_layer_call_and_return_conditional_losses_384421954e%&=¢:
3¢0
&#
input_3ÿÿÿÿÿÿÿÿÿ

p

 
ª "¢

0 
 ´
K__inference_sequential_2_layer_call_and_return_conditional_losses_384421991e%&=¢:
3¢0
&#
input_3ÿÿÿÿÿÿÿÿÿ

p 

 
ª "¢

0 
 ³
K__inference_sequential_2_layer_call_and_return_conditional_losses_384422076d%&<¢9
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
 ³
K__inference_sequential_2_layer_call_and_return_conditional_losses_384422113d%&<¢9
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
 
0__inference_sequential_2_layer_call_fn_384422008X%&=¢:
3¢0
&#
input_3ÿÿÿÿÿÿÿÿÿ

p

 
ª " 
0__inference_sequential_2_layer_call_fn_384422025X%&=¢:
3¢0
&#
input_3ÿÿÿÿÿÿÿÿÿ

p 

 
ª " 
0__inference_sequential_2_layer_call_fn_384422130W%&<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿ

p

 
ª " 
0__inference_sequential_2_layer_call_fn_384422147W%&<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿ

p 

 
ª " 
'__inference_signature_wrapper_384421903k%&7¢4
¢ 
-ª*
(
input_3
input_3 
"(ª%
#
dense_6
dense_6 