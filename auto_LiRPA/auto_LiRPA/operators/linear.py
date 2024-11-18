###########################################################################
##   This file is part of the auto_LiRPA library, a core part of the   ##
##   α,β-CROWN (alpha-beta-CROWN) neural network verifier developed    ##
##   by the α,β-CROWN Team                                             ##
##                                                                     ##
##   Copyright (C) 2020-2024 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com>                ##
##                     Zhouxing Shi <zshi@cs.ucla.edu>                 ##
##                     Kaidi Xu <kx46@drexel.edu>                      ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
###########################################################################
""" 
Module containing classes for handling linear and dot product layers 
with potential weight perturbations. These classes are essential for 
propagating bounds through neural network layers during verification.
"""
from torch import Tensor
from torch.nn import Module
from typing import Tuple, List
from .activation_base import BoundOptimizableActivation
from .base import *
from .bivariate import BoundMul, MulHelper
from .leaf import BoundParams
from ..patches import Patches, inplace_unfold
from .solver_utils import grb
from .clampmult import multiply_by_A_signs

EPS = 1e-2  # Small epsilon to prevent numerical issues in bounds

class BoundLinear(BoundOptimizableActivation):
    """
    Class representing a linear (fully connected) layer with potential 
    weight and bias perturbations. It handles bound propagation 
    (forward and backward) through the linear transformation.
    """
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        """
        Initializes the BoundLinear layer with attributes and options.
        
        Args:
            attr (dict): Attributes of the layer, such as transposition flags and scaling factors.
            inputs (list): Input nodes to this layer.
            output_index (int): Index of the output.
            options (dict): Additional options for the layer, like matmul optimizations.
        """
        # Initialize parent class
        super().__init__(attr, inputs, output_index, options)

        # Defaults based on ONNX specifications
        self.transA = 0  # Flag indicating if input A is transposed
        self.transB = 0  # Flag indicating if input B is transposed
        self.alpha_linear = 1.0  # Scaling factor for the linear transformation
        self.beta_linear = 1.0   # Scaling factor for the bias

        # Override defaults with provided attributes, if any
        if attr is not None:
            self.transA = attr.get('transA', self.transA)
            self.transB = attr.get('transB', self.transB)
            self.alpha_linear = attr.get('alpha', self.alpha_linear)
            self.beta_linear = attr.get('beta', self.beta_linear)

        # Additional options
        options = options or {}
        self.opt_matmul = options.get('matmul')  # MatMul optimization settings
        self.splittable = False  # Indicates if the layer can be split for parallel processing

        self.mul_helper = MulHelper()  # Helper for handling multiplications in bounds
        self.use_seperate_weights_for_lower_and_upper_bounds = False  # Flag for separate weights
        self.batched_weight_and_bias = False  # Flag indicating if weights and biases are batched
        self.share_alphas = options.get('matmul', {}).get('share_alphas', False)  # Flag for sharing alpha parameters

    def _preprocess(self, a, b, c=None):
        """
        Preprocesses inputs by handling transpositions and scaling factors.
        
        Args:
            a (Tensor): First input tensor (e.g., input activations).
            b (Tensor): Second input tensor (e.g., weights).
            c (Tensor, optional): Bias tensor.
        
        Returns:
            Tuple[Tensor, Tensor, Tensor]: Preprocessed tensors (a, b, c).
        """
        # Transpose input 'a' if required
        if self.transA and isinstance(a, Tensor):
            a = a.transpose(-2, -1)
        # Scale input 'a' by alpha_linear
        if self.alpha_linear != 1.0:
            a = self.alpha_linear * a
        # Transpose input 'b' if not already transposed
        if not self.transB and isinstance(b, Tensor):
            b = b.transpose(-2, -1)
        # Scale bias 'c' by beta_linear if present
        if c is not None:
            if self.beta_linear != 1.0:
                c = self.beta_linear * c
        return a, b, c

    def init_opt_parameters(self, start_nodes):
        """
        Initializes optimization parameters for the layer based on starting nodes.
        
        Args:
            start_nodes (list): List of starting nodes for bound propagation.
        """
        shared_alpha_dims = []
        if self.share_alphas:
            # Determine dimensions to share alpha parameters based on the number of matmul layers
            count_matmul = len([item for item in self._all_optimizable_activations
                                if isinstance(item, BoundLinear)])
            if count_matmul >= 6:
                shared_alpha_dims = [1, 2, 3]
            elif count_matmul >= 4:
                shared_alpha_dims = [1, 2]

        # Extract lower and upper bounds from input nodes
        input_lb = [xi.lower for xi in self.inputs]
        input_ub = [xi.upper for xi in self.inputs]
        # Preprocess the bounds
        input_lb = self._preprocess(*input_lb)
        input_ub = self._preprocess(*input_ub)
        # Reshape the bounds appropriately for matrix multiplication
        x_l, x_u, y_l, y_u = self._reshape(input_lb[0], input_ub[0], input_lb[1], input_ub[1])
        assert x_l.ndim == y_l.ndim
        # Determine the shape for alpha parameters based on shared dimensions
        shape = [1 if i in shared_alpha_dims
                 else max(x_l.shape[i], y_l.shape[i]) for i in range(x_l.ndim)]
        for start_node in start_nodes:
            ns, size_s = start_node[:2]
            # Initialize alpha parameters as ones
            if isinstance(size_s, torch.Size):
                size_s = prod(size_s)
            elif isinstance(size_s, (list, tuple)):
                size_s = size_s[0]
            self.alpha[ns] = torch.ones(4, size_s, *shape, device=x_l.device)

    def forward(self, x, w, b=None):
        """
        Forward pass of the linear layer without bound considerations.
        
        Args:
            x (Tensor): Input activations.
            w (Tensor): Weight matrix.
            b (Tensor, optional): Bias vector.
        
        Returns:
            Tensor: Output activations after linear transformation.
        """
        # Preprocess inputs
        x, w, b = self._preprocess(x, w, b)
        self.input_shape = self.x_shape = x.shape
        self.y_shape = w.t().shape
        # Perform matrix multiplication
        res = x.matmul(w.t())
        # Add bias if present
        if b is not None:
            res += b
        return res

    def onehot_mult(self, weight, bias, C, batch_size):
        """
        Multiplies the weight matrix with a diagonal matrix defined by C's selected rows.
        
        Args:
            weight (Tensor): Weight matrix.
            bias (Tensor): Bias vector.
            C (SomeType): Constraint matrix specifying selected rows and coefficients.
            batch_size (int): Number of samples in the batch.
        
        Returns:
            Tuple[Tensor, Tensor]: New weight matrix and bias vector after multiplication.
        """
        if C is None:
            return None, 0.0

        new_weight = None
        new_bias = 0.0

        # Transpose indices and coefficients if C has 2 dimensions
        if C.index.ndim == 2:
            index = C.index.transpose(0, 1)
            coeffs = C.coeffs.transpose(0, 1)
        else:
            index = C.index
            coeffs = C.coeffs

        if C.index.ndim == 1:
            # Every element in the batch shares the same rows
            if weight is not None:
                new_weight = self.non_deter_index_select(
                    weight, dim=0, index=index
                ).unsqueeze(1).expand(
                    [-1, batch_size] + [-1] * (weight.ndim - 1))
            if bias is not None:
                new_bias = self.non_deter_index_select(
                    bias, dim=0, index=index
                ).unsqueeze(1).expand(-1, batch_size)
        elif C.index.ndim == 2:
            # Each element in the batch has different rows
            if weight is not None:
                new_weight = batched_index_select(
                    weight.unsqueeze(0), dim=1, index=index)
            if bias is not None:
                new_bias = batched_index_select(
                    bias.unsqueeze(0), dim=1, index=index)
        
        # Apply coefficients if provided
        if C.coeffs is not None:
            if weight is not None:
                new_weight = new_weight * coeffs.unsqueeze(-1)
            if bias is not None:
                new_bias = new_bias * coeffs

        # Transpose back if C.index has 2 dimensions
        if C.index.ndim == 2:
            new_weight = new_weight.transpose(0, 1)
            new_bias = new_bias.transpose(0, 1)
        
        return new_weight, new_bias

    def bound_backward(self, last_lA, last_uA, *x, start_node=None,
                       reduce_bias=True, **kwargs):
        """
        Backward bound propagation through the linear layer.
        
        Args:
            last_lA (Tensor or Patches): Lower bound coefficients from the next layer.
            last_uA (Tensor or Patches): Upper bound coefficients from the next layer.
            *x: Inputs to the layer (input activations, weights, and optionally bias).
            start_node (Node, optional): Starting node for bound propagation.
            reduce_bias (bool): Whether to reduce bias terms.
            **kwargs: Additional keyword arguments.
        
        Returns:
            Tuple: Tuple containing bound matrices and bias terms.
        """
        assert len(x) == 2 or len(x) == 3  # Expecting input, weight, and optionally bias
        if start_node is not None:
            self._start = start_node.name
        has_bias = len(x) == 3  # Determine if bias is present

        # Extract lower and upper bounds from inputs
        input_lb = [xi.lower for xi in x]
        input_ub = [xi.upper for xi in x]

        # Preprocess the bounds (handle transpositions and scaling)
        input_lb = self._preprocess(*input_lb)
        input_ub = self._preprocess(*input_ub)

        lA_y = uA_y = lA_bias = uA_bias = None
        lbias = ubias = 0
        batch_size = last_lA.shape[1] if last_lA is not None else last_uA.shape[1]
        weight = input_lb[1]  # Extract weight from lower bounds
        bias = input_lb[2] if has_bias else None  # Extract bias if present

        def _bound_oneside(last_A, weight_override=None):
            """
            Helper function to compute bounds for one side (lower or upper).
            
            Args:
                last_A (Tensor or Patches): Bound coefficients from the next layer.
                weight_override (Tensor, optional): Override weight matrix for separate bounds.
            
            Returns:
                Tuple[Tensor, Tensor]: Next layer's bound coefficients and bias.
            """
            if weight_override is None:
                used_weight = weight
            else:
                used_weight = weight_override

            if last_A is None:
                return None, 0
            if isinstance(last_A, torch.Tensor):
                # Matrix mode: Multiply weight with bound matrices and compute bias
                if self.batched_weight_and_bias:
                    # Handle batched weights and biases
                    mod_last_A = last_A.unsqueeze(2)
                    mod_used_weight = used_weight.unsqueeze(0)
                    mod_next_A = mod_last_A.to(mod_used_weight).matmul(mod_used_weight)
                    next_A = mod_next_A.squeeze(2)
                    if has_bias:
                        mod_bias = bias.unsqueeze(0).unsqueeze(3)
                        mod_sum_bias = mod_last_A.to(mod_bias).matmul(mod_bias)
                        sum_bias = mod_sum_bias.squeeze(3).squeeze(2)
                else:
                    # Standard matrix multiplication
                    next_A = last_A.to(used_weight).matmul(used_weight)
                    sum_bias = (last_A.to(bias).matmul(bias)
                        if has_bias else 0.0)
            else:
                # Handle Patches mode for convolutional layers or similar structures
                assert isinstance(last_A, Patches)
                assert not self.batched_weight_and_bias
                reshaped_weight = used_weight.transpose(0, 1).view(
                    -1, *last_A.input_shape[1:])
                unfolded_weight = inplace_unfold(
                    reshaped_weight,
                    kernel_size=last_A.patches.shape[-2:],
                    stride=last_A.stride, padding=last_A.padding,
                    inserted_zeros=last_A.inserted_zeros,
                    output_padding=last_A.output_padding)
                if has_bias:
                    reshaped_bias = bias.view(*last_A.input_shape[1:]).unsqueeze(0)
                    unfolded_bias = inplace_unfold(
                        reshaped_bias, kernel_size=last_A.patches.shape[-2:],
                        stride=last_A.stride, padding=last_A.padding,
                        inserted_zeros=last_A.inserted_zeros,
                        output_padding=last_A.output_padding)
                if last_A.unstable_idx is not None:
                    selected_weight = unfolded_weight.permute(1, 2, 3, 4, 5, 0).unsqueeze(2)
                    next_A = torch.einsum('sbchw,sbchwi->sbi', last_A.patches, selected_weight)
                    if has_bias:
                        selected_bias = unfolded_bias.permute(1, 2, 0, 3, 4, 5)
                        sum_bias = torch.einsum('sbchw,sbchw->sb', last_A.patches, selected_bias)
                else:
                    selected_weight = unfolded_weight.permute(1, 2, 3, 4, 5, 0).unsqueeze(0).unsqueeze(0)
                    next_A_r = torch.einsum('sbpqchw,sbpqchwi->spqbi', last_A.patches, selected_weight)
                    next_A = next_A_r.reshape(-1, next_A_r.size(-2), next_A_r.size(-1))
                    if has_bias:
                        selected_bias = unfolded_bias.unsqueeze(0)
                        sum_bias_r = torch.einsum('sbpqchw,sbpqchw->spqb', last_A.patches, selected_bias)
                        sum_bias = sum_bias_r.reshape(-1, sum_bias_r.size(-1))
            return next_A, sum_bias if has_bias else 0.0

        # Case #1: No perturbation on weights or biases, only on inputs
        if not self.is_input_perturbed(1) and (not has_bias or not self.is_input_perturbed(2)):
            def multiply_with_weight(weight, set_l: bool, set_u: bool):
                """
                Multiplies the weight matrix with bound coefficients for lower and upper bounds.
                
                Args:
                    weight (Tensor): Weight matrix.
                    set_l (bool): Whether to set lower bound coefficients.
                    set_u (bool): Whether to set upper bound coefficients.
                
                Returns:
                    Tuple[Tensor, Tensor, float, float]: Updated lower and upper bound coefficients and biases.
                """
                lA_x = uA_x = None
                lbias = ubias = 0.
                if isinstance(last_lA, eyeC) and isinstance(last_uA, eyeC):
                    # If the previous layer has identity matrices, use current weights directly
                    shape_others = prod(last_lA.shape[2:-1])
                    A_identity = torch.eye(
                        shape_others, device=weight.device, dtype=weight.dtype
                    ).view(shape_others, 1, 1, shape_others, 1)
                    w = weight.view(
                        1, weight.size(0), *[1] * (len(last_lA.shape) - 2),
                        weight.size(1))
                    w = w * A_identity
                    tmp_A_x = w.reshape(
                        last_lA.shape[0], 1, *last_lA.shape[2:-1], weight.size(1)
                    ).expand(last_lA.shape[0], *last_lA.shape[1:-1], weight.size(1))
                    if set_l:
                        lA_x = tmp_A_x
                    if set_u:
                        uA_x = tmp_A_x
                    if has_bias:
                        tmp_bias = bias.unsqueeze(1).repeat(1, batch_size)
                        if set_l:
                            lbias = tmp_bias
                        if set_u:
                            ubias = tmp_bias
                elif isinstance(last_lA, OneHotC) or isinstance(last_uA, OneHotC):
                    # Handle OneHotC type bound coefficients by selecting specific rows
                    if set_l:
                        lA_x, lbias = self.onehot_mult(weight, bias, last_lA, batch_size)
                    if last_lA is last_uA and set_l and set_u:
                        uA_x = lA_x
                        ubias = lbias
                    elif set_u:
                        uA_x, ubias = self.onehot_mult(weight, bias, last_uA, batch_size)
                else:
                    # General case: multiply weights with last layer's bound coefficients
                    if set_l:
                        lA_x, lbias = _bound_oneside(last_lA, weight_override=weight)
                    if set_u:
                        uA_x, ubias = _bound_oneside(last_uA, weight_override=weight)
                return lA_x, uA_x, lbias, ubias

            if self.use_seperate_weights_for_lower_and_upper_bounds:
                # Use separate weights for lower and upper bounds
                lA_x, _, lbias, _ = multiply_with_weight(input_lb[1], set_l=True, set_u=False)
                _, uA_x, _, ubias = multiply_with_weight(input_ub[1], set_l=False, set_u=True)
            else:
                # Use the same weights for both lower and upper bounds
                lA_x, uA_x, lbias, ubias = multiply_with_weight(weight, set_l=True, set_u=True)

        # Case #2: Weights are perturbed; handle bound propagation with weight perturbation
        elif self.is_input_perturbed(1):
            assert not self.use_seperate_weights_for_lower_and_upper_bounds
            # Obtain relaxed bounds considering weight perturbations
            [(lA_x, uA_x), (lA_y, uA_y)], lbias, ubias = self.bound_backward_with_weight(
                last_lA, last_uA, input_lb, input_ub, x[0], x[1],
                reduce_bias=reduce_bias, **kwargs)
            if has_bias:
                assert reduce_bias
                if x[2].perturbation is not None:
                    # Bias is also perturbed; treat it as an additional input
                    lA_bias = last_lA
                    uA_bias = last_uA
                else:
                    # Bias is not perturbed; directly add it to the bias term
                    if isinstance(last_lA, eyeC) and isinstance(last_uA, eyeC):
                        lbias += input_lb[2].unsqueeze(1).repeat(1, batch_size)
                        ubias += input_lb[2].unsqueeze(1).repeat(1, batch_size)
                    else:
                        if last_lA is not None:
                            lbias += last_lA.matmul(input_lb[2])
                        if last_uA is not None:
                            ubias += last_uA.matmul(input_lb[2])
        # Case #3: Only bias is perturbed; weights are fixed
        elif not self.is_input_perturbed(1) and has_bias and self.is_input_perturbed(2):
            assert not self.use_seperate_weights_for_lower_and_upper_bounds
            assert reduce_bias
            if isinstance(last_lA, eyeC) and isinstance(last_uA, eyeC):
                # Use current weights directly when last layer has identity coefficients
                lA_x = uA_x = input_lb[1].unsqueeze(1).repeat(
                    [1, batch_size] + [1] * (input_lb[1].ndim - 1))
            else:
                # Multiply last layer's coefficients with current weights
                lA_x = last_lA.matmul(input_lb[1])
                uA_x = last_uA.matmul(input_lb[1])
            # Propagate bias coefficients
            lA_bias = last_lA
            uA_bias = last_uA
        else:
            # Ensure separate weight handling is not used in other cases
            assert not self.use_seperate_weights_for_lower_and_upper_bounds

        # Return the propagated bound coefficients and bias terms
        return [(lA_x, uA_x), (lA_y, uA_y), (lA_bias, uA_bias)], lbias, ubias

    def _reshape(self, x_l, x_u, y_l, y_u):
        """
        Reshapes the lower and upper bounds for proper matrix multiplication.
        
        Args:
            x_l (Tensor): Lower bound of input activations.
            x_u (Tensor): Upper bound of input activations.
            y_l (Tensor): Lower bound of weights.
            y_u (Tensor): Upper bound of weights.
        
        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: Reshaped lower and upper bounds.
        """
        x_shape, y_shape = self.input_shape, self.y_shape

        # Insert a dimension before the last for input activations
        x_l = x_l.unsqueeze(-2)
        x_u = x_u.unsqueeze(-2)

        # Handle reshaping based on the dimensions of x and y
        if len(x_shape) == len(y_shape):
            # Insert a dimension before the third last for weights
            y_l = y_l.unsqueeze(-3)
            y_u = y_u.unsqueeze(-3)
        elif len(y_shape) == 2:
            # Reshape weights to match input dimensions when y has 2 dimensions
            y_l = y_l.reshape(*([1] * (len(x_shape) - 2)), *y_shape).unsqueeze(-3)
            y_u = y_u.reshape(*([1] * (len(x_shape) - 2)), *y_shape).unsqueeze(-3)
        else:
            raise ValueError(f'Unsupported shapes: x_shape {x_shape}, y_shape {y_shape}')

        return x_l, x_u, y_l, y_u

    @staticmethod
    # @torch.jit.script
    def propagate_A_xy(last_A: Tensor, alpha_pos: Tensor, alpha_neg: Tensor,
                       beta_pos: Tensor, beta_neg: Tensor,
                       dim_y: List[int]) -> Tuple[Tensor, Tensor]:
        """
        Propagates bound coefficients through the linear transformation.
        
        Args:
            last_A (Tensor): Bound coefficients from the previous layer.
            alpha_pos (Tensor): Positive alpha parameters for bounds.
            alpha_neg (Tensor): Negative alpha parameters for bounds.
            beta_pos (Tensor): Positive beta parameters for bounds.
            beta_neg (Tensor): Negative beta parameters for bounds.
            dim_y (List[int]): Dimensions to sum over for output bounds.
        
        Returns:
            Tuple[Tensor, Tensor]: Propagated lower and upper bound coefficients.
        """
        # Clamp last_A into positive and negative parts
        last_A_pos = last_A.clamp(min=0).unsqueeze(-1)
        last_A_neg = last_A.clamp(max=0).unsqueeze(-1)
        # Compute A_x by multiplying alpha parameters with bound coefficients
        A_x = (alpha_pos.transpose(-1, -2).matmul(last_A_pos) +
               alpha_neg.transpose(-1, -2).matmul(last_A_neg)).squeeze(-1)
        # Compute A_y by multiplying beta parameters with bound coefficients
        A_y, _ = multiply_by_A_signs(last_A.unsqueeze(-1), beta_pos, beta_neg, None, None)
        if len(dim_y) != 0:
            A_y = torch.sum(A_y, dim=dim_y)
        return A_x, A_y

    def bound_backward_with_weight(self, last_lA, last_uA, input_lb, input_ub,
                                   x, y, reduce_bias=True, **kwargs):
        """
        Handles backward bound propagation when weights are perturbed.
        
        Args:
            last_lA (Tensor): Lower bound coefficients from the next layer.
            last_uA (Tensor): Upper bound coefficients from the next layer.
            input_lb (list): List of lower bounds for inputs.
            input_ub (list): List of upper bounds for inputs.
            x (Node): Input node activations.
            y (Node): Weight node.
            reduce_bias (bool): Whether to reduce bias terms.
            **kwargs: Additional keyword arguments.
        
        Returns:
            Tuple: Propagated bound coefficients and bias terms.
        """
        # Retrieve relaxation parameters for multiplication
        (alpha_l, beta_l, gamma_l,
         alpha_u, beta_u, gamma_u) = self.mul_helper.get_relaxation(
            *self._reshape(input_lb[0], input_ub[0], input_lb[1], input_ub[1]),
            self.opt_stage, getattr(self, 'alpha', None),
            getattr(self, '_start', None))
        x_shape = input_lb[0].size()
        if reduce_bias:
            # Sum gamma parameters over the last dimension if reducing bias
            gamma_l = torch.sum(gamma_l, dim=-1)
            gamma_u = torch.sum(gamma_u, dim=-1)

        # Determine dimensions to sum over for y based on input shapes
        if len(x.output_shape) != 2 and len(x.output_shape) == len(y.output_shape):
            dim_y = [-3]
        elif len(y.output_shape) == 2:
            dim_y = list(range(2, 2 + len(x_shape) - 2))
        else:
            raise NotImplementedError

        def _bound_oneside(last_A, alpha_pos, beta_pos, gamma_pos, alpha_neg, beta_neg, gamma_neg):
            """
            Helper function to compute propagated bounds for one side.
            
            Args:
                last_A (Tensor): Bound coefficients from the next layer.
                alpha_pos, beta_pos, gamma_pos (Tensor): Positive parameters.
                alpha_neg, beta_neg, gamma_neg (Tensor): Negative parameters.
            
            Returns:
                Tuple[Tensor, Tensor, Tensor]: Propagated A_x, A_y, and bias.
            """
            if last_A is None:
                return None, None, 0
            if isinstance(last_A, eyeC):
                # Handle identity coefficients
                last_A = (torch.eye(last_A.shape[0], device=last_A.device)
                    .view(last_A.shape[0], 1, *last_A.shape[2:]).expand(last_A.shape))

            # Propagate A_x and A_y using the helper function
            A_x, A_y = BoundLinear.propagate_A_xy(
                last_A, alpha_pos, alpha_neg, beta_pos, beta_neg, dim_y)

            if reduce_bias:
                # Compute bias based on the current stage
                if self.opt_stage in ['opt', 'reuse']:
                    bias = (torch.einsum('sb...,sb...->sb',
                                        last_A.clamp(min=0), gamma_pos)
                            + torch.einsum('sb...,sb...->sb',
                                        last_A.clamp(max=0), gamma_neg))
                else:
                    bias = (
                        self.get_bias(last_A.clamp(min=0), gamma_pos)
                        + self.get_bias(last_A.clamp(max=0), gamma_neg)
                    )
            else:
                # Handle bias without reduction
                assert self.batch_dim == 0
                assert self.opt_stage not in ['opt', 'reuse']
                assert dim_y == [-3]
                bias = (last_A.unsqueeze(-1).clamp(min=0) * gamma_pos
                        + last_A.unsqueeze(-1).clamp(max=0) * gamma_neg)
                bias_x = bias.sum(dim=-2)
                bias_y = bias.sum(dim=-3)
                bias = (bias_x, bias_y)
            return A_x, A_y, bias

        # Depending on the optimization stage, compute bounds differently
        if self.opt_stage in ['opt', 'reuse']:
            lA_x, lA_y, lbias = _bound_oneside(
                last_lA, alpha_l[0], beta_l[0], gamma_l[0],
                alpha_u[0], beta_u[0], gamma_u[0])
            uA_x, uA_y, ubias = _bound_oneside(
                last_uA, alpha_u[1], beta_u[1], gamma_u[1],
                alpha_l[1], beta_l[1], gamma_l[1])
        else:
            lA_x, lA_y, lbias = _bound_oneside(
                last_lA, alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u)
            uA_x, uA_y, ubias = _bound_oneside(
                last_uA, alpha_u, beta_u, gamma_u, alpha_l, beta_l, gamma_l)

        return [(lA_x, uA_x), (lA_y, uA_y)], lbias, ubias

    @staticmethod
    def _propagate_Linf(x, w):
        """
        Propagates bounds for L-infinity norm perturbations.
        
        Args:
            x (Tuple[Tensor, Tensor]): Tuple containing lower and upper bounds of inputs.
            w (Tensor): Weight matrix.
        
        Returns:
            Tuple[Tensor, Tensor]: Center and deviation for the output bounds.
        """
        h_L, h_U = x
        mid = (h_L + h_U) / 2  # Center of the interval
        diff = (h_U - h_L) / 2  # Half-width of the interval
        w_abs = w.abs()
        if mid.ndim == 2 and w.ndim == 3:
            center = torch.bmm(mid.unsqueeze(1), w.transpose(-1, -2)).squeeze(1)
            deviation = torch.bmm(diff.unsqueeze(1), w_abs.transpose(-1, -2)).squeeze(1)
        else:
            center = mid.matmul(w.transpose(-1, -2))
            deviation = diff.matmul(w_abs.transpose(-1, -2))
        return center, deviation

    def interval_propagate(self, *v, C=None, w=None):
        """
        Propagates interval bounds through the linear layer.
        
        Args:
            *v: Tuple containing bounds (and optionally bias).
            C (Tensor, optional): Constraint matrix for output transformation.
            w (Tensor, optional): External weight matrix for propagation.
        
        Returns:
            Tuple[Tensor, Tensor]: Lower and upper bounds after propagation.
        """
        has_bias = self is not None and len(v) == 3
        if self is not None:
            # Convert Interval objects to separate lower and upper bounds
            v_lb, v_ub = zip(*v)
            v_lb = self._preprocess(*v_lb)
            v_ub = self._preprocess(*v_ub)
            # Reconstruct Interval objects after preprocessing
            v = [Interval.make_interval(bounds[0], bounds[1], bounds[2])
                 for bounds in zip(v_lb, v_ub, v)]
        if w is None and self is None:
            # Use C as the weight with no bias
            w, lb, ub = C, torch.tensor(0., device=C.device), torch.tensor(0., device=C.device)
        else:
            if w is None:
                # Use the layer's weight, potentially perturbed
                if self.is_input_perturbed(1):  # Input index 1 corresponds to weights
                    assert C is None  # C matrix merging not supported with weight perturbation
                    res = self.interval_propagate_with_weight(*v)
                    l, u = res
                    if has_bias:
                        return l + v[2][0], u + v[2][1]
                    else:
                        return l, u
                else:
                    # Use fixed weight matrix from inputs
                    w = v[1][0]
            if has_bias:
                lb, ub = v[2]
            else:
                lb = ub = 0.0

            if C is not None:
                # Apply constraint matrix C to weights and biases
                w = C.matmul(w)
                lb = C.matmul(lb) if not isinstance(lb, float) else lb
                ub = C.matmul(ub) if not isinstance(ub, float) else ub

        # Handle different perturbation norms
        norm, eps = Interval.get_perturbation(v[0])[:2]
        if norm == torch.inf:
            # L-infinity norm perturbation
            interval = BoundLinear._propagate_Linf(v[0], w)
            center, deviation = interval
        elif norm > 0:
            # General Lp norm perturbation
            norm, eps = Interval.get_perturbation(v[0])
            mid = v[0][0]
            dual_norm = np.float64(1.0) / (1 - 1.0 / norm)
            if w.ndim == 3:
                # Handle batched weights
                center = w.matmul(mid.unsqueeze(-1)).squeeze(-1)
            else:
                center = mid.matmul(w.t())
            deviation = w.norm(dual_norm, dim=-1) * eps
        else:
            # Handle L0 norm perturbation
            norm, eps, ratio = Interval.get_perturbation(v[0])
            mid = v[0][0]
            weight_abs = w.abs()
            if w.ndim == 3:
                # Handle batched weights
                center = w.matmul(mid.unsqueeze(-1)).squeeze(-1)
            else:
                center = mid.matmul(w.t())
            k = int(eps)
            deviation = torch.sum(torch.topk(weight_abs, k)[0], dim=1) * ratio

        # Compute final lower and upper bounds
        lower, upper = center - deviation + lb, center + deviation + ub

        return (lower, upper)

    def interval_propagate_with_weight(self, *v):
        """
        Interval propagation when weights are perturbed.
        
        Args:
            *v: Tuple containing bounds (and optionally bias).
        
        Returns:
            Tuple[Tensor, Tensor]: Lower and upper bounds after propagation.
        """
        input_norm, input_eps = Interval.get_perturbation(v[0])
        weight_norm, weight_eps = Interval.get_perturbation(v[1])

        if input_norm == torch.inf and weight_norm == torch.inf:
            # Both input and weight have L-infinity norm perturbations
            if self.opt_matmul == 'economic':
                # Memory-efficient matmul implementation
                x_l, x_u = v[0][0], v[0][1]
                y_l, y_u = v[1][0].transpose(-1, -2), v[1][1].transpose(-1, -2)

                dx, dy = F.relu(x_u - x_l), F.relu(y_u - y_l)
                base = x_l.matmul(y_l)

                mask_xp, mask_xn = (x_l > 0).to(x_l.dtype), (x_u < 0).to(x_l.dtype)
                mask_xpn = 1 - mask_xp - mask_xn
                mask_yp, mask_yn = (y_l > 0).to(y_l.dtype), (y_u < 0).to(y_l.dtype)
                mask_ypn = 1 - mask_yp - mask_yn

                lower, upper = base.clone(), base.clone()

                lower += dx.matmul(y_l.clamp(max=0)) - (dx * mask_xn).matmul(y_l * mask_ypn)
                upper += dx.matmul(y_l.clamp(min=0)) + (dx * mask_xp).matmul(y_l * mask_ypn)

                lower += x_l.clamp(max=0).matmul(dy) - (x_l * mask_xpn).matmul(dy * mask_yn)
                upper += x_l.clamp(min=0).matmul(dy) + (x_l * mask_xpn).matmul(dy * mask_yp)

                lower += (dx * mask_xn).matmul(dy * mask_yn)
                upper += (dx * (mask_xpn + mask_xp)).matmul(dy * (mask_ypn + mask_yp))
            else:
                # General case: use bivariate bound propagation for perturbed inputs and weights
                x_l, x_u = v[0][0].unsqueeze(-2), v[0][1].unsqueeze(-2)
                y_l, y_u = v[1][0].unsqueeze(-3), v[1][1].unsqueeze(-3)
                lower, upper = BoundMul.interval_propagate_both_perturbed(*[(x_l, x_u), (y_l, y_u)])
                lower, upper = torch.sum(lower, -1), torch.sum(upper, -1)

            return lower, upper
        elif input_norm == torch.inf and weight_norm == 2:
            # Input has L-infinity norm and weight has L2 norm perturbations
            eps = weight_eps
            h_L, h_U = v[0]
            center, deviation = BoundLinear._propagate_Linf(v[0], v[1][0])
            max_l2 = torch.max(h_L.abs(), h_U.abs()).norm(2, dim=-1).unsqueeze(-1)
            lb, ub = center - deviation - max_l2 * eps, center + deviation + max_l2 * eps
            return lb, ub
        else:
            # Unsupported perturbation combination
            raise NotImplementedError(
                "Unsupported perturbation combination: data={}, weight={}".format(input_norm, weight_norm))

    @staticmethod
    @torch.jit.script
    def bound_forward_mul(x_lw: Tensor, x_lb: Tensor, x_uw: Tensor, x_ub: Tensor, w: Tensor):
        """
        Computes forward bounds for multiplication.
        
        Args:
            x_lw (Tensor): Lower bound of linear weights.
            x_lb (Tensor): Lower bound of inputs.
            x_uw (Tensor): Upper bound of linear weights.
            x_ub (Tensor): Upper bound of inputs.
            w (Tensor): Weight matrix.
        
        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: Lower and upper bounds for both lower and upper parts.
        """
        w_pos, w_neg = w.clamp(min=0), w.clamp(max=0)
        lw = x_lw.matmul(w_pos) + x_uw.matmul(w_neg)
        uw = x_uw.matmul(w_pos) + x_lw.matmul(w_neg)
        lb = x_lb.matmul(w_pos) + x_ub.matmul(w_neg)
        ub = x_ub.matmul(w_pos) + x_lb.matmul(w_neg)
        return lw, lb, uw, ub

    def bound_dynamic_forward(self, x, w=None, b=None, C=None, max_dim=None, offset=0):
        """
        Dynamically computes forward bounds, typically used in optimization solvers.
        
        Args:
            x (Interval): Interval bounds of inputs.
            w (Tensor, optional): External weight matrix.
            b (Tensor, optional): External bias vector.
            C (Tensor, optional): Constraint matrix.
            max_dim (int, optional): Maximum dimension for processing.
            offset (int, optional): Offset for variable indexing.
        
        Returns:
            LinearBound: Object containing the computed lower and upper bounds.
        """
        assert not self.transA and self.alpha_linear == 1.0 and self.transB and self.beta_linear == 1.0
        assert not self.is_input_perturbed(1)
        assert not self.is_input_perturbed(2)

        weight = w.lb
        bias = b.lb if b is not None else None
        if C is not None:
            # Merge specification C into last layer weights
            weight = C.squeeze(0).mm(weight)
            if bias is not None:
                bias = C.squeeze(0).mm(bias.unsqueeze(-1)).view(-1)
            lb = x.lb.unsqueeze(1)
        else:
            weight = weight.t()
            lb = x.lb

        # Compute new lower and upper weights and biases
        w_new = x.lw.matmul(weight)
        b_new = lb.matmul(weight)
        if C is not None:
            b_new = b_new.squeeze(1)
        if bias is not None:
            b_new += bias

        return LinearBound(w_new, b_new, w_new, b_new, x_L=x.x_L, x_U=x.x_U, tot_dim=x.tot_dim)

    def bound_forward(self, dim_in, x, w=None, b=None, C=None):
        """
        Computes forward bounds for the linear layer.
        
        Args:
            dim_in (int): Input dimension.
            x (LinearBound): Object containing input bounds.
            w (LinearBound, optional): External weight bounds.
            b (LinearBound, optional): External bias bounds.
            C (Tensor, optional): Constraint matrix.
        
        Returns:
            LinearBound: Object containing the computed lower and upper bounds.
        """
        has_bias = b is not None
        # Preprocess inputs (handle transpositions and scaling)
        x, w, b = self._preprocess(x, w, b)

        # Case #1: No perturbation on weights or biases, only on inputs
        if not self.is_input_perturbed(1) and (not has_bias or not self.is_input_perturbed(2)):
            if isinstance(w, LinearBound):
                w = w.lower
            if isinstance(b, LinearBound):
                b = b.lower
            if C is not None:
                # Apply constraint matrix C to weights and biases
                w = C.to(w).matmul(w).transpose(-1, -2)
                if b is not None:
                    b = C.to(b).matmul(b)
                x_lb, x_ub = x.lb.unsqueeze(1), x.ub.unsqueeze(1)
            else:
                # Transpose weights for matrix multiplication
                w = w.t()
                x_lb, x_ub = x.lb, x.ub
            # Compute lower and upper bounds using the bound_forward_mul static method
            lw, lb, uw, ub = BoundLinear.bound_forward_mul(x.lw, x_lb, x.uw, x_ub, w)

            if C is not None:
                lb, ub = lb.squeeze(1), ub.squeeze(1)

            if b is not None:
                lb += b
                ub += b

        # Case #2: Weights are perturbed; handle bound propagation with weight perturbation
        elif self.is_input_perturbed(1):
            if C is not None:
                raise NotImplementedError
            res = self.bound_forward_with_weight(dim_in, x, w)
            if has_bias:
                raise NotImplementedError
            lw, lb, uw, ub = res.lw, res.lb, res.uw, res.ub

        # Case #3: Only bias is perturbed; weights are fixed
        elif not self.is_input_perturbed(1) and has_bias and self.is_input_perturbed(2):
            raise NotImplementedError

        # Return the computed lower and upper bounds encapsulated in a LinearBound object
        return LinearBound(lw, lb, uw, ub)

    def bound_forward_with_weight(self, dim_in, x, y):
        """
        Handles forward bound propagation when weights are perturbed.
        
        Args:
            dim_in (int): Input dimension.
            x (LinearBound): Object containing input bounds.
            y (LinearBound): Object containing weight bounds.
        
        Returns:
            LinearBound: Object containing the computed lower and upper bounds.
        """
        # Unsqueeze dimensions for matrix multiplication
        x_unsqueeze = LinearBound(
            x.lw.unsqueeze(-2), x.lb.unsqueeze(-2),
            x.uw.unsqueeze(-2), x.ub.unsqueeze(-2),
            x.lower.unsqueeze(-2), x.upper.unsqueeze(-2),
        )
        y_unsqueeze = LinearBound(
            y.lw.unsqueeze(-3), y.lb.unsqueeze(-3),
            y.uw.unsqueeze(-3), y.ub.unsqueeze(-3),
            y.lower.unsqueeze(-3), y.upper.unsqueeze(-3),
        )
        # Perform bound propagation with both inputs perturbed
        res_mul = self.bound_forward_both_perturbed(dim_in, x_unsqueeze, y_unsqueeze)
        # Sum over the appropriate dimension to obtain final bounds
        return LinearBound(
            res_mul.lw.sum(dim=-1) if res_mul.lw is not None else None,
            res_mul.lb.sum(dim=-1),
            res_mul.uw.sum(dim=-1) if res_mul.uw is not None else None,
            res_mul.ub.sum(dim=-1)
        )

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        """
        Constructs solver variables and constraints for optimization-based verification.
        
        Args:
            *v: Tuple containing variables from the previous layer.
            model (gurobi.Model): Optimization model to add variables and constraints.
            C (Tensor, optional): Constraint matrix.
            model_type (str): Type of the model (e.g., "mip").
            solver_pkg (str): Solver package to use (e.g., "gurobi").
        """
        has_bias = self is not None and len(v) == 3
        gvars_array = np.array(v[0])  # Array of Gurobi variables from the previous layer
        pre_layer_shape = gvars_array.shape  # Shape of the previous layer
        this_layer_shape = self.lower.squeeze(0).shape  # Shape of the current layer's output
        out_lbs = self.lower.squeeze(0).detach().cpu().numpy() if self.lower is not None else None
        out_ubs = self.upper.squeeze(0).detach().cpu().numpy() if self.upper is not None else None

        # Extract weight matrix and transpose if necessary
        this_layer_weight = v[1]
        if self.transB == 0:
            this_layer_weight = this_layer_weight.transpose(1, 0)
        if C is not None:
            # Apply constraint matrix C to merge specifications
            this_layer_weight = C.squeeze(0).mm(this_layer_weight)
        this_layer_weight = this_layer_weight.detach().cpu().numpy()

        this_layer_bias = None
        if has_bias:
            # Extract and potentially transform bias vector
            this_layer_bias = v[2]
            if C is not None:
                this_layer_bias = C.squeeze(0).mm(this_layer_bias.unsqueeze(-1)).view(-1)
            this_layer_bias = this_layer_bias.detach().cpu().numpy()

        new_layer_gurobi_vars = []

        # Iterate over each neuron in the current layer to create solver variables
        for neuron_idx in range(this_layer_shape[0]):
            out_lb = out_lbs[neuron_idx] if out_lbs is not None else -float('inf')
            out_ub = out_ubs[neuron_idx] if out_ubs is not None else float('inf')
            if out_lbs is not None and out_ubs is not None:
                # Adjust bounds to prevent floating point discrepancies
                diff = out_ub - out_lb
                avg = (out_ub + out_lb) / 2.0
                condition = (diff < EPS)
                out_lb = np.where(condition, avg - EPS / 2.0, out_lb)
                out_ub = np.where(condition, avg + EPS / 2.0, out_ub)

            lin_expr = 0
            if has_bias:
                lin_expr = this_layer_bias[neuron_idx].item()
            coeffs = this_layer_weight[neuron_idx, :]

            if solver_pkg == 'gurobi':
                # Create a linear expression using Gurobi's LinExpr
                lin_expr += grb.LinExpr(coeffs, v[0])
            else:
                # Fallback for other solvers (inefficient)
                for i in range(len(coeffs)):
                    try:
                        lin_expr += coeffs[i] * v[0][i]
                    except TypeError:
                        lin_expr += coeffs[i] * v[0][i].var

            # Add a new variable for the current neuron with specified bounds
            var = model.addVar(lb=out_lb, ub=out_ub, obj=0,
                                vtype=grb.GRB.CONTINUOUS,
                                name=f'lay{self.name}_{neuron_idx}')
            # Add equality constraint linking the linear expression to the new variable
            model.addConstr(lin_expr == var, name=f'lay{self.name}_{neuron_idx}_eq')
            new_layer_gurobi_vars.append(var)

        self.solver_vars = new_layer_gurobi_vars  # Store the solver variables
        model.update()

    def build_gradient_node(self, grad_upstream):
        """
        Constructs gradient nodes for backpropagation.
        
        Args:
            grad_upstream (Tensor): Gradient from the upstream layer.
        
        Returns:
            List[Tuple[Module, Tuple, List]]: Gradient modules and their inputs.
        """
        if not self.inputs[1].from_input:
            if isinstance(self.inputs[1], BoundParams):
                w = self.inputs[1].param
            else:
                w = self.inputs[1].value
            if not self.transB:
                w = w.t()
            node_grad = LinearGrad(w.detach())
            return [(node_grad, (grad_upstream,), [])]
        else:
            assert not self.transB
            w = self.inputs[1].forward_value
            node_grad = MatMulGrad()
            return [
                (node_grad, (grad_upstream, self.inputs[1].forward_value), []),
                (node_grad, (grad_upstream, self.inputs[0].forward_value), []),
            ]

    def update_requires_input_bounds(self):
        """
        Updates the requirements for input bounds based on weight perturbations.
        """
        self._check_weight_perturbation()


class BoundMatMul(BoundLinear):
    """
    Class representing a matrix multiplication layer. Inherits from BoundLinear
    and reuses most of its functionalities with slight modifications.
    """
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        """
        Initializes the BoundMatMul layer.
        
        Args:
            attr (dict): Attributes of the layer.
            inputs (list): Input nodes to this layer.
            output_index (int): Index of the output.
            options (dict): Additional options for the layer.
        """
        super().__init__(attr, inputs, output_index, options)
        self.transA = 0  # No transposition for input A by default
        self.transB = 0  # No transposition for input B by default
        self.splittable = True  # Indicates that the layer can be split

    def forward(self, x, y):
        """
        Forward pass for matrix multiplication.
        
        Args:
            x (Tensor): First input matrix.
            y (Tensor): Second input matrix.
        
        Returns:
            Tensor: Result of matrix multiplication.
        """
        self.x_shape = x.shape
        self.y_shape = y.shape
        return x.matmul(y)

    def interval_propagate(self, *v):
        """
        Interval bound propagation for matrix multiplication.
        
        Args:
            *v: Tuple containing bounds.
        
        Returns:
            Tuple[Tensor, Tensor]: Lower and upper bounds after propagation.
        """
        lower, upper = super().interval_propagate(*v)
        return lower, upper

    def bound_backward(self, last_lA, last_uA, *x, start_node=None, **kwargs):
        """
        Backward bound propagation through matrix multiplication.
        
        Args:
            last_lA (Tensor): Lower bound coefficients from the next layer.
            last_uA (Tensor): Upper bound coefficients from the next layer.
            *x: Inputs to the layer (matrix inputs).
            start_node (Node, optional): Starting node for bound propagation.
            **kwargs: Additional keyword arguments.
        
        Returns:
            Tuple: Tuple containing bound matrices and bias terms.
        """
        assert len(x) == 2  # Expecting two inputs for matrix multiplication
        if start_node is not None:
            self._start = start_node.name
        # Use parent class's backward bound propagation
        results = super().bound_backward(last_lA, last_uA, *x, **kwargs)
        # Transpose the bound coefficients for the second input
        lA_y = results[0][1][0].transpose(-1, -2) if results[0][1][0] is not None else None
        uA_y = results[0][1][1].transpose(-1, -2) if results[0][1][1] is not None else None
        # Handle bias terms
        if isinstance(results[1], tuple):
            lbias = (results[1][0], results[1][1].transpose(-1, -2))
        else:
            lbias = results[1]
        if isinstance(results[2], tuple):
            ubias = (results[2][0], results[2][1].transpose(-1, -2))
        else:
            ubias = results[2]
        return [results[0][0], (lA_y, uA_y), results[0][2]], lbias, ubias

    def bound_forward(self, dim_in, x, y):
        """
        Computes forward bounds for matrix multiplication.
        
        Args:
            dim_in (int): Input dimension.
            x (LinearBound): Object containing input bounds.
            y (LinearBound): Object containing weight bounds.
        
        Returns:
            LinearBound: Object containing the computed lower and upper bounds.
        """
        return super().bound_forward(dim_in, x, LinearBound(
            y.lw.transpose(-1, -2) if y.lw is not None else None,
            y.lb.transpose(-1, -2) if y.lb is not None else None,
            y.uw.transpose(-1, -2) if y.uw is not None else None,
            y.ub.transpose(-1, -2) if y.ub is not None else None,
            y.lower.transpose(-1, -2) if y.lower is not None else None,
            y.upper.transpose(-1, -2) if y.upper is not None else None
        ))

    def update_requires_input_bounds(self):
        """
        Updates the requirements for input bounds based on whether the second input is perturbed.
        """
        # Check if the second input (weights) is not perturbed, indicating linear operation
        self.is_linear_op = not self.inputs[1].perturbed
        if self.is_linear_op:
            # If one input is constant, no bounds are required on it
            self.requires_input_bounds = []
            self.splittable = False
        else:
            # If both inputs are perturbed, bounds are required for both
            self.requires_input_bounds = [0, 1]
            self.splittable = True


class BoundNeg(Bound):
    """
    Class representing a negation operation in the network.
    """
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        """
        Initializes the BoundNeg layer.
        
        Args:
            attr (dict): Attributes of the layer.
            inputs (list): Input nodes to this layer.
            output_index (int): Index of the output.
            options (dict): Additional options for the layer.
        """
        super().__init__(attr, inputs, output_index, options)
        self.ibp_intermediate = True  # Indicates that this layer can be used in IBP (Interval Bound Propagation)

    def forward(self, x):
        """
        Forward pass that negates the input.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Negated input.
        """
        return -x

    def bound_backward(self, last_lA, last_uA, x, **kwargs):
        """
        Backward bound propagation for negation.
        
        Args:
            last_lA (Tensor or Patches): Lower bound coefficients from the next layer.
            last_uA (Tensor or Patches): Upper bound coefficients from the next layer.
            x (Node): Input node.
            **kwargs: Additional keyword arguments.
        
        Returns:
            Tuple: Tuple containing bound coefficients and bias terms.
        """
        if type(last_lA) == Tensor or type(last_uA) == Tensor:
            # Negate the bound coefficients if they are tensors
            return [(-last_lA if last_lA is not None else None,
                     -last_uA if last_uA is not None else None)], 0, 0
        elif type(last_lA) == Patches or type(last_uA) == Patches:
            # Negate the patches if they are in Patches mode
            if last_lA is not None:
                lA = Patches(-last_lA.patches, last_lA.stride, last_lA.padding,
                             last_lA.shape, unstable_idx=last_lA.unstable_idx,
                             output_shape=last_lA.output_shape)
            else:
                lA = None

            if last_uA is not None:
                uA = Patches(-last_uA.patches, last_uA.stride, last_uA.padding,
                             last_uA.shape, unstable_idx=last_uA.unstable_idx,
                             output_shape=last_uA.output_shape)
            else:
                uA = None
            return [(lA, uA)], 0, 0
        else:
            raise NotImplementedError

    def bound_forward(self, dim_in, x):
        """
        Computes forward bounds for negation.
        
        Args:
            dim_in (int): Input dimension.
            x (LinearBound): Object containing input bounds.
        
        Returns:
            LinearBound: Object containing the computed lower and upper bounds.
        """
        return LinearBound(-x.uw, -x.ub, -x.lw, -x.lb)

    def interval_propagate(self, *v):
        """
        Interval bound propagation for negation.
        
        Args:
            *v: Tuple containing bounds.
        
        Returns:
            Tuple[Tensor, Tensor]: Lower and upper bounds after negation.
        """
        return -v[0][1], -v[0][0]


class BoundCumSum(Bound):
    """
    Class representing a cumulative sum operation in the network.
    """
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        """
        Initializes the BoundCumSum layer.
        
        Args:
            attr (dict): Attributes of the layer.
            inputs (list): Input nodes to this layer.
            output_index (int): Index of the output.
            options (dict): Additional options for the layer.
        """
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = True  # Indicates that default IBP can be used

    def forward(self, x, axis):
        """
        Forward pass that computes the cumulative sum along a specified axis.
        
        Args:
            x (Tensor): Input tensor.
            axis (int): Axis along which to compute the cumulative sum.
        
        Returns:
            Tensor: Cumulative sum of the input tensor.
        """
        self.axis = axis
        return torch.cumsum(x, axis)


class BoundIdentity(Bound):
    """
    Class representing an identity operation in the network.
    """
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        """
        Initializes the BoundIdentity layer.
        
        Args:
            attr (dict): Attributes of the layer.
            inputs (list): Input nodes to this layer.
            output_index (int): Index of the output.
            options (dict): Additional options for the layer.
        """
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = True  # Indicates that default IBP can be used

    def forward(self, x):
        """
        Forward pass that returns the input as-is.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Same as input tensor.
        """
        return x

    def bound_backward(self, last_lA, last_uA, x, **kwargs):
        """
        Backward bound propagation for identity operation.
        
        Args:
            last_lA (Tensor or Patches): Lower bound coefficients from the next layer.
            last_uA (Tensor or Patches): Upper bound coefficients from the next layer.
            x (Node): Input node.
            **kwargs: Additional keyword arguments.
        
        Returns:
            Tuple: Tuple containing bound coefficients and bias terms.
        """
        return [(last_lA, last_uA)], 0, 0

    def bound_forward(self, dim_in, x):
        """
        Computes forward bounds for identity operation.
        
        Args:
            dim_in (int): Input dimension.
            x (LinearBound): Object containing input bounds.
        
        Returns:
            LinearBound: Object containing the same lower and upper bounds as input.
        """
        return x


class LinearGrad(Module):
    """
    Module representing the gradient computation for a linear layer during backpropagation.
    """
    def __init__(self, weight):
        """
        Initializes the LinearGrad module.
        
        Args:
            weight (Tensor): Weight matrix of the linear layer.
        """
        super().__init__()
        self.weight = weight

    def forward(self, grad_last):
        """
        Computes the gradient with respect to the input activations.
        
        Args:
            grad_last (Tensor): Gradient from the next layer.
        
        Returns:
            Tensor: Gradient with respect to the input activations.
        """
        weight = self.weight.to(grad_last).t()  # Transpose weight for gradient computation
        return F.linear(grad_last, weight)


class MatMulGrad(Module):
    """
    Module representing the gradient computation for a matrix multiplication layer during backpropagation.
    """
    def forward(self, grad_last, x):
        """
        Computes the gradient with respect to the inputs of the matrix multiplication.
        
        Args:
            grad_last (Tensor): Gradient from the next layer.
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Gradient with respect to the input tensor.
        """
        return grad_last.matmul(x.transpose(-1, -2))
