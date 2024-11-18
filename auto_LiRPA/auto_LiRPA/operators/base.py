#########################################################################
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
#########################################################################
""" Base class and functions for implementing bound operators"""

# Import necessary modules and libraries
from typing import Optional, List
import warnings
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

# Import custom modules from the auto_LiRPA package
from ..perturbations import *
from ..utils import *
from ..patches import *
from ..linear_bound import LinearBound

# Disable JIT profiling to potentially improve performance
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

# Define a small epsilon value to prevent division by zero or similar issues
epsilon = 1e-12

def not_implemented_op(node, func):
    """
    Raises a NotImplementedError with a standardized message when a function is not implemented.

    Args:
        node (Bound): The node for which the function is not implemented.
        func (str): The name of the function that is not implemented.

    Raises:
        NotImplementedError: Indicates that the function is not supported yet.
    """
    message = (
        f'Function `{func}` of `{node}` is not supported yet.'
        ' Please help to open an issue at https://github.com/Verified-Intelligence/auto_LiRPA'
        ' or implement this function in auto_LiRPA/bound_ops.py'
        ' or auto_LiRPA/operators by yourself.'
    )
    raise NotImplementedError(message)

class Interval(tuple):
    """ 
    Interval object for interval bound propagation.

    Inherits from tuple to leverage immutability and tuple functionalities.

    Attributes:
        ptb (Perturbation or None): The perturbation associated with the interval. 
            If None, the interval is treated as a constant with lb = ub.
    """

    def __new__(self, lb=None, ub=None, ptb=None):
        """
        Create a new Interval instance.

        Args:
            lb (Tensor or None): Lower bound of the interval.
            ub (Tensor or None): Upper bound of the interval.
            ptb (Perturbation or None): Perturbation associated with the interval.

        Returns:
            Interval: A new Interval instance containing (lb, ub).
        """
        return tuple.__new__(Interval, (lb, ub))

    def __init__(self, lb, ub, ptb=None):
        """
        Initialize the Interval instance.

        Args:
            lb (Tensor or None): Lower bound of the interval.
            ub (Tensor or None): Upper bound of the interval.
            ptb (Perturbation or None): Perturbation associated with the interval.
        """
        if ptb is None:
            self.ptb = None
            # `self.ptb == None` implies that this interval is not perturbed and lb = ub.
            # Ensure that lb and ub are the same object to maintain consistency.
            assert lb is ub, "For non-perturbed intervals, lb and ub must be the same."
        else:
            if not isinstance(ptb, Perturbation):
                raise ValueError("ptb must be a Perturbation object or None. Got type {}".format(type(ptb)))
            else:
                self.ptb = ptb

    def __str__(self):
        """Return a human-readable string representation of the Interval."""
        return "({}, {}) with ptb={}".format(self[0], self[1], self.ptb)

    def __repr__(self):
        """Return an unambiguous string representation of the Interval."""
        return "Interval(lb={}, ub={}, ptb={})".format(self[0], self[1], self.ptb)

    @staticmethod
    def make_interval(lb, ub, other=None):
        """
        Create an Interval object, optionally preserving perturbation information.

        Args:
            lb (Tensor): Lower bound.
            ub (Tensor): Upper bound.
            other (Interval or tuple, optional): Another Interval or tuple to inherit perturbation from.

        Returns:
            Interval or tuple: A new Interval object with (lb, ub) and inherited perturbation if applicable.
        """
        if isinstance(other, Interval):
            return Interval(lb, ub, ptb=other.ptb)
        else:
            return (lb, ub)

    @staticmethod
    def get_perturbation(interval):
        """
        Retrieve the norm and epsilon values from an Interval's perturbation.

        Args:
            interval (Interval or tuple): The interval from which to extract perturbation info.

        Returns:
            tuple: A tuple containing (norm, eps) or additional parameters for certain perturbations.

        Raises:
            RuntimeError: If the perturbation type is unrecognized.
        """
        if isinstance(interval, Interval) and interval.ptb is not None:
            if isinstance(interval.ptb, PerturbationLpNorm):
                return interval.ptb.norm, interval.ptb.eps
            elif isinstance(interval.ptb, PerturbationSynonym):
                return torch.inf, 1.0
            elif isinstance(interval.ptb, PerturbationLpNormLocalised):
                return interval.ptb.norm, interval.ptb.eps, interval.ptb.window_size
            elif isinstance(interval.ptb, PerturbationL0Norm):
                return 0, interval.ptb.eps, interval.ptb.ratio
            else:
                raise RuntimeError("get_perturbation() does not know how to handle {}".format(type(interval.ptb)))
        else:
            # If no perturbation, assume L-infinity norm with default epsilon
            return torch.inf, np.nan

    @staticmethod
    def is_perturbed(interval):
        """
        Check if an Interval or tuple has perturbation enabled.

        Args:
            interval (Interval or tuple): The interval to check.

        Returns:
            bool: True if perturbed, False otherwise.
        """
        if isinstance(interval, Interval) and interval.ptb is None:
            return False
        else:
            return True

class Bound(nn.Module):
    r"""
    Base class for supporting the bound computation of an operator.

    This class serves as the foundational component for all bound operators in the auto_LiRPA library.
    It defines the interface and common functionalities required for bound computations.

    Args:
        attr (dict, optional): Attributes of the operator. Defaults to None.
        inputs (list, optional): A list of input nodes. Defaults to None.
        output_index (int, optional): The index in the output if the operator has multiple outputs. Defaults to 0.
        options (dict, optional): Bound options. Defaults to None.

    Notes:
        - Be sure to run `super().__init__(attr, inputs, output_index, options)` first in the `__init__` function.
        - This class should be subclassed to implement specific bound operators (e.g., BoundReLU, BoundLinear).
    """

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        """
        Initialize the Bound instance.

        Args:
            attr (dict, optional): Attributes of the operator. Defaults to None.
            inputs (list, optional): A list of input nodes. Defaults to None.
            output_index (int, optional): The index in the output if the operator has multiple outputs. Defaults to 0.
            options (dict, optional): Bound options. Defaults to None.
        """
        super().__init__()
        attr = {} if attr is None else attr  # Initialize attr as empty dict if None
        inputs = [] if inputs is None else inputs  # Initialize inputs as empty list if None
        options = {} if options is None else options  # Initialize options as empty dict if None

        # Basic attributes
        self.name: Optional[str] = None  # Name of the node/operator
        self.output_name = []  # List to store output names if multiple outputs
        self.device = attr.get('device')  # Device (CPU/GPU) information

        self.attr = attr  # Store operator attributes
        self.inputs: List['Bound'] = inputs  # List of input nodes
        self.output_index = output_index  # Index for multi-output operators
        self.options = options  # Store bound options

        self.forward_value = None  # Placeholder for storing forward pass values
        self.output_shape = None  # Shape of the operator's output
        self.from_input = False  # Flag indicating if this node is directly from input

        self.bounded = False  # Flag indicating if bounds have been computed
        self.IBP_rets = None  # Placeholder for storing Intermediate Bounds Propagation results

        self.requires_input_bounds = []  # List of input indices that require bounds

        # Flags for Jacobian handling
        self.no_jacobian = False  # If True, do not propagate Jacobian through this node

        # Flags for intermediate bounds handling
        self.ibp_intermediate = False  # If True, use IBP for intermediate bounds instead of CROWN

        self.splittable = False  # If True, the operator can be split for parallel bound computation

        # Perturbation flags
        self.perturbed = False  # Indicates if this node has a perturbed output
        self.never_perturbed = False  # If True, this node will never be perturbed

        # Loss fusion flag
        if options is not None and 'loss_fusion' in options:
            self.loss_fusion = options['loss_fusion']
        else:
            self.loss_fusion = False

        self.options = options  # Store bound options again (redundant, can be removed)

        # Default interval propagation flag
        self.use_default_ibp = False  # If True, use the default interval propagation method

        # Flags to zero out backward coefficients
        self.zero_backward_coeffs_l = False  # If True, set lower A matrix to zero
        self.zero_backward_coeffs_u = False  # If True, set upper A matrix to zero

        # Patch-based flag
        self.patches_start = False  # Indicates if patch-based bound computation starts at this node

        # Mask for alpha-beta optimization
        self.alpha_beta_update_mask = None  # Mask used in alpha-beta optimization

        # Final node flag
        self.is_final_node = False  # Indicates if this node is the final output node

        # Batch dimension index (default -1 means no batch dimension)
        self.batch_dim = -1  # To be updated in BoundedModule.get_forward_value()

        # Flags and caches for bound propagation
        self._is_lower_bound_current = False  # Indicates if the lower bound is up-to-date
        self._lower = None  # Cached lower bound tensor

        self._is_upper_bound_current = False  # Indicates if the upper bound is up-to-date
        self._upper = None  # Cached upper bound tensor

    def __repr__(self, attrs=None):
        """
        Return a string representation of the Bound instance.

        Args:
            attrs (dict, optional): Additional attributes to include in the representation.

        Returns:
            str: String representation of the Bound instance.
        """
        inputs = ', '.join([node.name for node in self.inputs])  # Get input node names
        ret = (f'{self.__class__.__name__}(name={self.name}, '
               f'inputs=[{inputs}], perturbed={self.perturbed}')
        if attrs is not None:
            for k, v in attrs.items():
                ret += f', {k}={v}'
        ret += ')'
        return ret

    @property
    def lower(self):
        """Get the lower bound."""
        return self._lower

    @lower.setter
    def lower(self, value):
        """
        Set the lower bound.

        Args:
            value (Tensor or None): The new lower bound tensor.

        Raises:
            TypeError: If the provided value is not a Tensor or None.
        """
        if not (value is None or isinstance(value, torch.Tensor)):
            raise TypeError(f'lower must be a tensor or None, got {type(value)}')
        if value is None:
            self._is_lower_bound_current = False
        else:
            self._is_lower_bound_current = True
        self._lower = value

    @property
    def upper(self):
        """Get the upper bound."""
        return self._upper

    @upper.setter
    def upper(self, value):
        """
        Set the upper bound.

        Args:
            value (Tensor or None): The new upper bound tensor.

        Raises:
            TypeError: If the provided value is not a Tensor or None.
        """
        if not (value is None or isinstance(value, torch.Tensor)):
            raise TypeError(f'upper must be a tensor or None, got {type(value)}')
        if value is None:
            self._is_upper_bound_current = False
        else:
            self._is_upper_bound_current = True
        self._upper = value

    def move_lower_and_upper_bounds_to_cache(self):
        """
        Move the current lower and upper bounds to the cache by detaching them and disabling gradients.

        This is useful for preventing gradients from flowing back through bound computations.
        """
        if self._lower is not None:
            self._lower = self._lower.detach().requires_grad_(False)
            self._is_lower_bound_current = False
        if self._upper is not None:
            self._upper = self._upper.detach().requires_grad_(False)
            self._is_upper_bound_current = False

    def delete_lower_and_upper_bounds(self):
        """
        Delete the cached lower and upper bounds.

        This can help save memory when bounds are no longer needed.
        """
        self._lower = None
        self._upper = None
        self._is_lower_bound_current = False
        self._is_upper_bound_current = False

    def is_lower_bound_current(self):
        """
        Check if the lower bound is up-to-date.

        Returns:
            bool: True if the lower bound is current, False otherwise.
        """
        return self._is_lower_bound_current

    def is_upper_bound_current(self):
        """
        Check if the upper bound is up-to-date.

        Returns:
            bool: True if the upper bound is current, False otherwise.
        """
        return self._is_upper_bound_current

    def are_output_constraints_activated_for_layer(
        self: 'Bound',
        apply_output_constraints_to: Optional[List[str]],
    ):
        """
        Determine if output constraints are activated for this layer based on provided criteria.

        Args:
            apply_output_constraints_to (List[str] or None): Layers to apply output constraints to.
                Each entry can be a layer name (starting with '/') or a layer type (starting with 'Bound').

        Returns:
            bool: True if output constraints are activated for this layer, False otherwise.
        """
        if self.is_final_node:
            return False  # Do not apply output constraints to the final node
        if apply_output_constraints_to is None:
            return False  # No layers specified for output constraints
        for layer_type_or_name in apply_output_constraints_to:
            if layer_type_or_name.startswith('/'):
                if self.name == layer_type_or_name:
                    return True  # Exact layer name match
            else:
                assert layer_type_or_name.startswith('Bound'), (
                    'To apply output constraints to tighten layer bounds, pass either the layer name '
                    '(starting with "/", e.g. "/input.7") or the layer type (starting with "Bound", '
                    'e.g. "BoundLinear")'
                )
                if type(self).__name__ == layer_type_or_name:
                    return True  # Layer type match
        return False  # No matches found

    def init_gammas(self, num_constraints):
        """
        Initialize gamma variables for alpha-beta optimization.

        Gammas are used in alpha-beta optimization to refine bounds.

        Args:
            num_constraints (int): The number of constraints for which to initialize gammas.
        """
        if not self.are_output_constraints_activated_for_layer(
            self.options.get('optimize_bound_args', {}).get('apply_output_constraints_to', [])
        ):
            return  # Do not initialize gammas if output constraints are not activated

        assert len(self.output_shape) > 0, self  # Ensure output shape is defined
        neurons_in_this_layer = 1
        for d in self.output_shape[1:]:
            neurons_in_this_layer *= d  # Calculate total neurons in the layer

        init_gamma_value = 0.0  # Initial value for gammas

        # Determine if gammas are shared across neurons
        if self.options.get('optimize_bound_args', {}).get('share_gammas', False):
            # Shared gammas across all neurons in the layer
            self.gammas_underlying_tensor = torch.full(
                (2, num_constraints, 1), init_gamma_value, requires_grad=True, device=self.device
            )
            self.gammas = self.gammas_underlying_tensor.expand(-1, -1, neurons_in_this_layer)
        else:
            # Separate gammas for each neuron
            self.gammas_underlying_tensor = torch.full(
                (2, num_constraints, neurons_in_this_layer), init_gamma_value, requires_grad=True, device=self.device
            )
            self.gammas = self.gammas_underlying_tensor

    def clip_gammas(self):
        """
        Clip gamma variables to ensure they remain non-negative.

        This is essential as gamma variables represent weights and should not be negative.
        """
        if not hasattr(self, "gammas"):
            return  # Exit if gammas are not initialized

        self.gammas_underlying_tensor.data = torch.clamp(
            self.gammas_underlying_tensor.data, min=0.0
        )  # Clamp to non-negative values

        # If gammas are shared, ensure that the expanded view reflects the clamped underlying tensor
        neurons_in_this_layer = 1
        for d in self.output_shape[1:]:
            neurons_in_this_layer *= d
        if self.options.get('optimize_bound_args', {}).get('share_gammas', False):
            self.gammas = self.gammas_underlying_tensor.expand(-1, -1, neurons_in_this_layer)

    def is_input_perturbed(self, i=0):
        """
        Check if the i-th input is perturbed.

        Args:
            i (int, optional): Index of the input to check. Defaults to 0.

        Returns:
            bool: True if the i-th input is perturbed, False otherwise.
        """
        return i < len(self.inputs) and self.inputs[i].perturbed

    def clear(self):
        """
        Clear attributes when there is a new input to the network.

        This function can be overridden by subclasses to reset or clear specific attributes.
        """
        pass

    @property
    def input_name(self):
        """
        Get the list of input node names.

        Returns:
            list: Names of the input nodes.
        """
        return [node.name for node in self.inputs]

    def forward(self, *x):
        """
        Function for standard/clean forward pass.

        Args:
            *x: A list of input values. The length of the list is equal to the number of input nodes.

        Raises:
            NotImplementedError: Indicates that the forward function is not implemented for this node.
        """
        return not_implemented_op(self, 'forward')

    def interval_propagate(self, *v):
        """
        Function for interval bound propagation (IBP) computation.

        By default, it uses `self.default_interval_propagate(*v)` if the operator is monotonic.
        Subclasses can override this method to implement custom IBP behaviors.

        Args:
            *v: A list of the interval bounds of input nodes.
                For each element `v[i]`, `v[i][0]` is the lower interval bound,
                and `v[i][1]` is the upper interval bound.

        Returns:
            Interval: The interval bound of this node, in the same format as v[i].

        Raises:
            NotImplementedError: If the operator does not implement interval propagation.
        """
        if self.use_default_ibp or self.never_perturbed:
            return self.default_interval_propagate(*v)
        else:
            return not_implemented_op(self, 'interval_propagate')

    def default_interval_propagate(self, *v):
        """
        Default IBP using the forward function.

        Suitable for unary monotonous functions or functions that alter shapes only without changing values.

        Args:
            *v: A list of input intervals.

        Returns:
            Interval: The resulting interval after applying the operator.
        """
        if len(v) == 0:
            # No inputs; assume a constant operator
            return Interval.make_interval(self.forward(), self.forward())
        else:
            if len(v) > 1:
                for i in range(1, len(v)):
                    assert not self.is_input_perturbed(i), "Only the first input can be perturbed."
            # Apply the forward function to the lower and upper bounds separately
            return Interval.make_interval(
                self.forward(v[0][0], *[vv[0] for vv in v[1:]]),
                self.forward(v[0][1], *[vv[0] for vv in v[1:]]), v[0]
            )

    def bound_forward(self, dim_in, *x):
        r"""
        Function for forward mode bound propagation.

        Forward mode LiRPA computes a `LinearBound` instance representing the linear bound for each involved node.
        Major attributes of `LinearBound` include `lw`, `uw`, `lb`, `ub`, `lower`, and `upper`.

        - `lw` and `uw` are coefficients of linear bounds with respect to model input.
          Their shape is `(batch_size, dim_in, *standard_shape)`,
          where `dim_in` is the total dimension of perturbed input nodes of the model,
          and `standard_shape` is the shape of the standard/clean output.
        - `lb` and `ub` are bias terms of linear bounds, and their shape is equal
          to the shape of standard/clean output.
        - `lower` and `upper` are concretized lower and upper bounds that will be
          computed later in BoundedModule.

        Args:
            dim_in (int): Total dimension of perturbed input nodes of the model.
            *x: A list of the linear bounds of input nodes. Each element in x is a `LinearBound` instance.

        Raises:
            NotImplementedError: If the bound_forward function is not implemented for this node.

        Returns:
            LinearBound: The linear bound of this node.
        """
        return not_implemented_op(self, 'bound_forward')

    def bound_dynamic_forward(self, *x, max_dim=None, offset=0):
        """
        Function for dynamic forward mode bound propagation.

        This can be used for operators that require dynamic handling during forward bound computation.

        Args:
            *x: Input nodes.
            max_dim (int, optional): Maximum dimension to consider. Defaults to None.
            offset (int, optional): Offset value. Defaults to 0.

        Raises:
            NotImplementedError: If the bound_dynamic_forward function is not implemented for this node.
        """
        raise NotImplementedError(f'bound_dynamic_forward is not implemented for {self}.')

    def bound_backward(self, last_lA, last_uA, *x, **kwargs):
        r"""
        Function for backward mode bound propagation.

        Backward mode LiRPA computes the bounds by propagating A matrices (linear coefficients) backward through the network.

        Args:
            last_lA (Tensor or None): `A` matrix for lower bound computation propagated to this node.
                It can be `None` if lower bound is not needed.
            last_uA (Tensor or None): `A` matrix for upper bound computation propagated to this node.
                It can be `None` if upper bound is not needed.
            *x: A list of input nodes, with `x[i].lower` and `x[i].upper` that can be used as pre-activation bounds.
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: If the bound_backward function is not implemented for this node.

        Returns:
            tuple: 
                - A (list): A list of A matrices for the input nodes. Each element is a tuple `(lA, uA)`.
                - lbias (Tensor): The bias term for lower bound computation, introduced by the linear relaxation of this node.
                - ubias (Tensor): The bias term for upper bound computation, introduced by the linear relaxation of this node.
        """
        return not_implemented_op(self, 'bound_backward')

    def broadcast_backward(self, A, x):
        """
        Adjust the shape of A by adding or removing broadcast dimensions based on the other operand x.

        Typically, A has shape `[spec, batch, ...]`.
        The other operand x may have shape `[batch, ...]`, or no batch dimension.
        The "..." dimensions may be different.
        This function ensures that the two shapes are compatible by summing over mismatched dimensions in A.

        Args:
            A (Tensor or Patches): The A matrix to be adjusted.
            x (Bound): The other operand to align shapes with.

        Returns:
            Tensor or Patches: The adjusted A matrix with compatible shape.
        """
        shape = x.output_shape  # Get the output shape of the other operand

        if isinstance(A, Tensor):
            if x.batch_dim == -1:
                # The other operand has no batch dimension; sum over extra dimensions in A
                shape = torch.Size([A.shape[1]] + list(shape))
                dims = []
                cnt_sum = A.ndim - len(shape) - 1
                for i in range(2, A.ndim):  # Iterate over dimensions to sum
                    if cnt_sum > 0:
                        dims.append(i)
                        cnt_sum -= 1
                if dims:
                    A = torch.sum(A, dim=dims)
            else:
                # Sum over extra dimensions in A that do not match the other operand's shape
                dims = list(range(1, A.ndim - len(shape)))
                if dims:
                    A = torch.sum(A, dim=dims)
            dims = []
            for i in range(1, len(shape)):
                # Identify dimensions where A needs to be summed to match the other operand
                if shape[i] == 1 and A.shape[i + 1] != 1:
                    dims.append(i + 1)
            if dims:
                A = torch.sum(A, dim=dims, keepdim=True)
            # Ensure the final shape matches by asserting
            assert A.shape[2:] == shape[1:], "After broadcasting, A's shape does not match the target shape."
        else:
            # If A is not a Tensor (e.g., Patches), additional handling can be implemented here
            pass
        return A

    def build_gradient_node(self, grad_upstream):
        """
        Function for building the gradient node to bound the Jacobian.

        This function should be implemented by subclasses to handle gradient computations specific to the operator.

        Args:
            grad_upstream: Upstream gradient in the gradient back-propagation.

        Raises:
            NotImplementedError: If the build_gradient_node function is not implemented for this node.
        """
        return not_implemented_op(self, 'build_gradient_node')

    def get_bias(self, A, bias):
        """
        Compute the bias term introduced by the linear relaxation of this node.

        Args:
            A (Tensor or Patches or None): The A matrix representing linear coefficients.
            bias (Tensor): The forward pass bias value.

        Returns:
            Tensor or int: The computed bias term. Returns 0 if A is None.

        Raises:
            NotImplementedError: If A is of an unsupported type.
        """
        if A is None:
            return 0  # No bias if A matrix is not provided
        if not Benchmarking:
            assert not isnan(A), "A matrix contains NaN values."
            assert not isnan(bias), "Bias contains NaN values."
        if torch.isinf(bias).any():
            warnings.warn('There is an inf value in the bias of LiRPA bounds.')

        if isinstance(A, Tensor):
            if self.batch_dim != -1:
                # If there's a batch dimension, perform matrix multiplication accordingly
                bias_new = torch.einsum('sb...,b...->sb', A, bias)
            else:
                # If no batch dimension, adjust einsum accordingly
                bias_new = torch.einsum('sb...,...->sb', A, bias)
            if isnan(bias_new):
                # Handle NaN cases, potentially caused by 0 * inf
                return 0
            else:
                return bias_new
        elif isinstance(A, eyeC):
            batch_size = A.shape[1]
            if self.batch_dim != -1:
                # Reshape bias for operations involving eyeC
                return bias.reshape(batch_size, -1).t()
            else:
                # Repeat bias across the batch dimension
                return bias.reshape(-1).unsqueeze(-1).repeat(1, batch_size)
        elif type(A) == Patches:
            # Handle Patches type A matrices
            if self.batch_dim != -1:
                # For patch-based operators with batch dimensions
                patches = A.patches  # Shape: [batch, L, out_c, in_c, K, K]
                bias = inplace_unfold(
                    bias, 
                    kernel_size=A.patches.shape[-2:], 
                    stride=A.stride, 
                    padding=A.padding, 
                    inserted_zeros=A.inserted_zeros, 
                    output_padding=A.output_padding
                )
                if A.unstable_idx is not None:
                    # Sparse bias handling
                    bias = bias[:, A.unstable_idx[1], A.unstable_idx[2]]
                    bias_new = torch.einsum('bschw,sbchw->sb', bias, patches)
                else:
                    # Dense bias handling
                    bias_new = torch.einsum('bijchw,sbijchw->sbij', bias, patches)
            else:
                # Handle cases without batch dimensions (e.g., BoundConstant)
                patches = A.patches
                bias_new = torch.sum(patches, dim=(-1, -2, -3)) * bias.to(self.device)
            return bias_new
        else:
            raise NotImplementedError("Unsupported type for A in get_bias.")

    def make_axis_non_negative(self, axis, shape='input'):
        """
        Convert a possibly negative axis index to a non-negative index based on the shape.

        Args:
            axis (int or list or tuple): The axis index or list of axis indices to convert.
            shape (str or torch.Size, optional): The shape context ('input' or 'output') or a specific torch.Size. Defaults to 'input'.

        Returns:
            int or list or tuple: The converted non-negative axis index/indices.
        """
        if isinstance(axis, (tuple, list)):
            # Recursively convert each axis in a tuple or list
            return tuple([self.make_axis_non_negative(item, shape) for item in axis])
        if shape == 'input':
            shape = self.input_shape
        elif shape == 'output':
            shape = self.output_shape
        else:
            assert isinstance(shape, torch.Size), "Shape must be 'input', 'output', or a torch.Size."
        if axis < 0:
            return axis + len(shape)  # Convert negative index to positive
        else:
            return axis

    def update_requires_input_bounds(self):
        """
        Update the list of input indices that require bounds.

        This function can be overridden by subclasses to specify which inputs need bound computations.
        """
        pass

    def clamp_interim_bounds(self):
        """
        Clamp intermediate bounds to ensure they remain within feasible ranges.

        This function can be overridden by subclasses to implement specific clamping strategies.
        """
        pass

    def check_constraint_available(self, node, flag=False):
        """
        Check if constraint intervals are available for a given node and its inputs.

        Args:
            node (Bound): The node to check.
            flag (bool, optional): Accumulator flag. Defaults to False.

        Returns:
            bool: True if constraints are available, False otherwise.
        """
        if hasattr(node, 'cstr_interval'):
            flag = True  # Constraint interval exists for this node
        for n in node.inputs:
            if not n.from_input:
                # Recursively check constraints for input nodes
                flag = flag or self.check_constraint_available(n, flag)
        return flag

    def _ibp_constraint(self, node: 'Bound', delete_bounds_after_use=False):
        """
        Internal function to propagate constraints using Interval Bound Propagation (IBP).

        Args:
            node (Bound): The node for which to compute constraints.
            delete_bounds_after_use (bool, optional): If True, delete unused bounds after computation. Defaults to False.

        Returns:
            tuple: The constraint interval (lower, upper) for the node.
        """
        def _delete_unused_bounds(node_list):
            """
            Delete bounds from input layers after use to save memory.

            Args:
                node_list (list): List of nodes whose bounds should be deleted.
            """
            if delete_bounds_after_use:
                for n in node_list:
                    del n.cstr_interval
                    del n.cstr_lower
                    del n.cstr_upper

        if not node.perturbed and hasattr(node, 'forward_value'):
            # If the node is not perturbed and has a forward value, set constraints as the forward value
            node.cstr_lower, node.cstr_upper = node.cstr_interval = (node.forward_value, node.forward_value)

        to_be_deleted_bounds = []
        if not hasattr(node, 'cstr_interval'):
            # If constraints are not already computed for this node, compute them
            for n in node.inputs:
                if not hasattr(n, 'cstr_interval'):
                    # Recursively compute constraints for input nodes
                    self._ibp_constraint(n, delete_bounds_after_use=delete_bounds_after_use)
                    to_be_deleted_bounds.append(n)
            # Collect the constraint intervals from input nodes
            inp = [n_pre.cstr_interval for n_pre in node.inputs]
            # Propagate constraints through the operator
            node.cstr_interval = node.interval_propagate(*inp)

            node.cstr_lower, node.cstr_upper = node.cstr_interval
            if isinstance(node.cstr_lower, torch.Size):
                node.cstr_lower = torch.tensor(node.cstr_lower)
                node.cstr_interval = (node.cstr_lower, node.cstr_upper)
            if isinstance(node.cstr_upper, torch.Size):
                node.cstr_upper = torch.tensor(node.cstr_upper)
                node.cstr_interval = (node.cstr_lower, node.cstr_upper)

        if node.is_lower_bound_current():
            # Update the node's bounds based on the constraint intervals
            node.lower = torch.where(
                node.lower >= node.cstr_lower, node.lower, node.cstr_lower
            )
            node.upper = torch.where(
                node.upper <= node.cstr_upper, node.upper, node.cstr_upper
            )
            node.interval = (node.lower, node.upper)

        # Delete unused bounds to free memory
        _delete_unused_bounds(to_be_deleted_bounds)
        return node.cstr_interval

    def _check_weight_perturbation(self):
        """
        Check if any of the input nodes (excluding the first one) have perturbations.

        This is typically used to determine if weight perturbations are present.

        Returns:
            bool: True if weight perturbations are present, False otherwise.
        """
        weight_perturbation = False
        for n in self.inputs[1:]:
            if hasattr(n, 'perturbation'):
                if n.perturbation is not None:
                    weight_perturbation = True
        if weight_perturbation:
            # If weight perturbations are present, all input indices are required
            self.requires_input_bounds = list(range(len(self.inputs)))
        else:
            # Otherwise, no additional inputs require bounds
            self.requires_input_bounds = []
        return weight_perturbation

    def non_deter_wrapper(self, op, *args, **kwargs):
        """
        Wrapper to handle non-deterministic operations by temporarily disabling deterministic algorithms.

        Some operations are non-deterministic and can cause failures in deterministic mode.
        This wrapper ensures that such operations run correctly by disabling deterministic algorithms.

        Args:
            op (callable): The operation to execute.
            *args: Positional arguments for the operation.
            **kwargs: Keyword arguments for the operation.

        Returns:
            The result of the operation.
        """
        if self.options.get('deterministic', False):
            # Disable deterministic algorithms temporarily
            torch.use_deterministic_algorithms(False)
        ret = op(*args, **kwargs)  # Execute the operation
        if self.options.get('deterministic', False):
            # Re-enable deterministic algorithms
            torch.use_deterministic_algorithms(True)
        return ret

    def non_deter_scatter_add(self, *args, **kwargs):
        """
        Wrapper for the torch.scatter_add operation to handle non-determinism.

        Args:
            *args: Positional arguments for torch.scatter_add.
            **kwargs: Keyword arguments for torch.scatter_add.

        Returns:
            Tensor: The result of torch.scatter_add.
        """
        return self.non_deter_wrapper(torch.scatter_add, *args, **kwargs)

    def non_deter_index_select(self, *args, **kwargs):
        """
        Wrapper for the torch.index_select operation to handle non-determinism.

        Args:
            *args: Positional arguments for torch.index_select.
            **kwargs: Keyword arguments for torch.index_select.

        Returns:
            Tensor: The result of torch.index_select.
        """
        return self.non_deter_wrapper(torch.index_select, *args, **kwargs)
