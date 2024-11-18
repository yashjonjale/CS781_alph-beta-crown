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

import os
import torch
from torch import Tensor
from collections import deque
from tqdm import tqdm
from .patches import Patches
from .utils import *
from .bound_ops import *
import warnings

from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from .bound_general import BoundedModule

def batched_backward(self: 'BoundedModule', node, C, unstable_idx, batch_size,
                     bound_lower=True, bound_upper=True, return_A=None):
    """
    Perform batched backward bound propagation (CROWN) to handle large numbers of unstable neurons.

    Args:
        self (BoundedModule): The bounded module instance.
        node (Bound): The node for which bounds are being computed.
        C (str or torch.Tensor): Specification matrix or a string indicating the type ('OneHot', 'Patches', etc.).
        unstable_idx (torch.Tensor or tuple): Indices of unstable neurons.
        batch_size (int): Number of samples in the batch.
        bound_lower (bool, optional): If True, compute lower bounds. Defaults to True.
        bound_upper (bool, optional): If True, compute upper bounds. Defaults to True.
        return_A (bool, optional): If True, return linear coefficients (A tensors). Defaults to None.

    Returns:
        tuple: Contains lower bounds, upper bounds, and optionally the A dictionary.
    """
    if return_A is None:
        return_A = self.return_A  # Use the instance's return_A flag if not specified

    crown_batch_size = self.bound_opts['crown_batch_size']  # Maximum batch size for CROWN
    output_shape = node.output_shape[1:]  # Shape of the node's output excluding batch dimension
    dim = int(prod(output_shape))  # Total number of output neurons

    # Determine if the indexing is dense or sparse
    if unstable_idx is None:
        unstable_idx = torch.arange(dim, device=self.device)
        dense = True
    else:
        dense = False

    unstable_size = get_unstable_size(unstable_idx)  # Number of unstable neurons
    print(f'Batched CROWN: node {node}, unstable size {unstable_size}')
    num_batches = (unstable_size + crown_batch_size - 1) // crown_batch_size  # Calculate number of batches
    ret = []  # List to store results
    ret_A = {}  # Dictionary to store A matrices if return_A is True

    for i in tqdm(range(num_batches), desc="Processing Batches"):
        # Slice the unstable indices for the current batch
        if isinstance(unstable_idx, tuple):
            unstable_idx_batch = tuple(
                u[i*crown_batch_size:(i+1)*crown_batch_size]
                for u in unstable_idx
            )
            unstable_size_batch = len(unstable_idx_batch[0])
        else:
            unstable_idx_batch = unstable_idx[i*crown_batch_size:(i+1)*crown_batch_size]
            unstable_size_batch = len(unstable_idx_batch)

        # Create the specification matrix for the current batch
        if node.patches_start and node.mode == "patches":
            assert C in ['Patches', None], "C must be 'Patches' or None for patch-based nodes."
            C_batch = Patches(
                shape=[
                    unstable_size_batch, batch_size, *node.output_shape[1:-2], 1, 1
                ],
                identity=1,
                unstable_idx=unstable_idx_batch,
                output_shape=[batch_size, *node.output_shape[1:]]
            )
        elif isinstance(node, (BoundLinear, BoundMatMul)):
            assert C in ['OneHot', None], "C must be 'OneHot' or None for linear or MatMul nodes."
            C_batch = OneHotC(
                [batch_size, unstable_size_batch, *node.output_shape[1:]],
                self.device,
                unstable_idx_batch,
                None
            )
        else:
            assert C in ['eye', None], "C must be 'eye' or None for other node types."
            C_batch = torch.zeros([1, unstable_size_batch, dim], device=self.device)
            C_batch[0, torch.arange(unstable_size_batch), unstable_idx_batch] = 1.0
            C_batch = C_batch.expand(batch_size, -1, -1).view(
                batch_size, unstable_size_batch, *output_shape
            )

        # Temporarily override the return_A option for this batch
        ori_return_A_option = self.return_A
        self.return_A = return_A

        # Perform backward bound computation for the current batch
        batch_ret = self.backward_general(
            node, C_batch,
            bound_lower=bound_lower, bound_upper=bound_upper,
            average_A=False, need_A_only=False, unstable_idx=unstable_idx_batch,
            verbose=False
        )
        ret.append(batch_ret[:2])  # Append lower and upper bounds

        if len(batch_ret) > 2:
            # If A matrices are returned, merge them into ret_A
            batch_A = batch_ret[2]
            ret_A = merge_A(batch_A, ret_A)

        # Restore the original return_A option
        self.return_A = ori_return_A_option

    # Concatenate the results from all batches
    if bound_lower:
        lb = torch.cat([item[0].view(batch_size, -1) for item in ret], dim=1)
        if dense:
            # Reshape to the original output shape if dense
            lb = lb.reshape(batch_size, *output_shape)
    else:
        lb = None

    if bound_upper:
        ub = torch.cat([item[1].view(batch_size, -1) for item in ret], dim=1)
        if dense:
            # Reshape to the original output shape if dense
            ub = ub.reshape(batch_size, *output_shape)
    else:
        ub = None

    # Return the results based on the return_A flag
    if return_A:
        return lb, ub, ret_A
    else:
        return lb, ub


def backward_general(
    self: 'BoundedModule',
    bound_node,
    C,
    start_backpropagation_at_node=None,
    bound_lower=True,
    bound_upper=True,
    average_A=False,
    need_A_only=False,
    unstable_idx=None,
    update_mask=None,
    verbose=True,
    apply_output_constraints_to: Optional[List[str]] = None,
    initial_As: Optional[dict] = None,
    initial_lb: Optional[torch.tensor] = None,
    initial_ub: Optional[torch.tensor] = None,
):
    """
    General backward bound propagation method to compute bounds for a given node.

    Args:
        self (BoundedModule): The bounded module instance.
        bound_node (Bound): The node for which bounds are being computed.
        C (str or torch.Tensor): Specification matrix or a string indicating the type.
        start_backpropagation_at_node (Bound, optional): Node where backpropagation starts.
        bound_lower (bool, optional): If True, compute lower bounds. Defaults to True.
        bound_upper (bool, optional): If True, compute upper bounds. Defaults to True.
        average_A (bool, optional): If True, average the A matrices. Defaults to False.
        need_A_only (bool, optional): If True, only compute A matrices without bounds. Defaults to False.
        unstable_idx (torch.Tensor or tuple, optional): Indices of unstable neurons.
        update_mask (torch.Tensor, optional): Mask for updating bounds.
        verbose (bool, optional): If True, enable verbose logging. Defaults to True.
        apply_output_constraints_to (List[str], optional): Layers to apply output constraints.
        initial_As (dict, optional): Initial A matrices for specific layers.
        initial_lb (torch.Tensor, optional): Initial lower bound tensor.
        initial_ub (torch.Tensor, optional): Initial upper bound tensor.

    Returns:
        tuple: Contains lower bounds, upper bounds, and optionally the A dictionary.
    """
    use_beta_crown = self.bound_opts['optimize_bound_args']['enable_beta_crown']  # Flag for beta-CROWN
    best_of_oc_and_no_oc = (
        self.bound_opts['optimize_bound_args']['best_of_oc_and_no_oc']
    )  # Flag to choose the best bounds between with and without output constraints
    tighten_input_bounds = (
        self.bound_opts['optimize_bound_args']['tighten_input_bounds']
    )  # Flag to tighten input bounds

    # Initialize the infeasible_bounds tensor if not already done
    if self.infeasible_bounds is None:
        device = bound_node.attr['device']
        if isinstance(C, Patches):
            self.infeasible_bounds = torch.full((C.shape[1],), False, device=device)
        else:
            assert isinstance(C, (torch.Tensor, eyeC, OneHotC)), type(C)
            self.infeasible_bounds = torch.full((C.shape[0],), False, device=device)

    # Check if output constraints are activated for the layer
    if bound_node.are_output_constraints_activated_for_layer(apply_output_constraints_to):
        assert not use_beta_crown, "Beta-CROWN is not compatible with output constraints."
        assert not self.cut_used, "Cutting is not compatible with output constraints."
        assert initial_As is None, "Initial A matrices should not be provided with output constraints."
        assert initial_lb is None, "Initial lower bound should not be provided with output constraints."
        assert initial_ub is None, "Initial upper bound should not be provided with output constraints."

        if best_of_oc_and_no_oc:
            # Compute bounds without output constraints first
            with torch.no_grad():
                o_res = self.backward_general(
                    bound_node=bound_node,
                    C=C,
                    start_backpropagation_at_node=start_backpropagation_at_node,
                    bound_lower=bound_lower,
                    bound_upper=bound_upper,
                    average_A=average_A,
                    need_A_only=need_A_only,
                    unstable_idx=unstable_idx,
                    update_mask=update_mask,
                    verbose=verbose,
                    apply_output_constraints_to=[],
                )
        # Compute bounds with output constraints
        res = self.backward_general_with_output_constraint(
            bound_node=bound_node,
            C=C,
            start_backporpagation_at_node=start_backpropagation_at_node,
            bound_lower=bound_lower,
            bound_upper=bound_upper,
            average_A=average_A,
            need_A_only=need_A_only,
            unstable_idx=unstable_idx,
            update_mask=update_mask,
            verbose=verbose,
        )

        if best_of_oc_and_no_oc:
            # Choose the best bounds between with and without output constraints
            res0_inf_mask = torch.isinf(res[0])
            r0 = res[0] - res[0].detach() + torch.max(res[0].detach(), o_res[0].detach())
            r0 = torch.where(res0_inf_mask, res[0], r0)

            res1_inf_mask = torch.isinf(res[1])
            r1 = res[1] - res[1].detach() + torch.min(res[1].detach(), o_res[1].detach())
            r1 = torch.where(res1_inf_mask, res[1], r1)

            if self.return_A:
                if res[2] != {}:
                    raise NotImplementedError(
                        "Merging of A not implemented yet. If set, try disabling --best_of_oc_and_no_oc"
                    )
                res = (r0, r1, {})
            else:
                res = (r0, r1)

        batch_size = res[0].size(0)
        infeasible_bounds = torch.any(res[0].reshape((batch_size, -1)) > res[1].reshape((batch_size, -1)), dim=1)
        if torch.any(infeasible_bounds):
            # Update infeasible_bounds if any bounds are unsatisfiable
            self.infeasible_bounds = torch.logical_or(self.infeasible_bounds, infeasible_bounds)
        return res

    # Retrieve the root nodes (input nodes)
    roots = self.roots()

    if start_backpropagation_at_node is None:
        # Determine the starting node for backpropagation
        start_backpropagation_at_node = bound_node

    if verbose:
        # Log information about the backpropagation process
        logger.debug(f'Bound backward from {start_backpropagation_at_node.__class__.__name__}({start_backpropagation_at_node.name}) '
                     f'to bound {bound_node.__class__.__name__}({bound_node.name})')
        if isinstance(C, str):
            logger.debug(f'  C: {C}')
        elif C is not None:
            logger.debug(f'  C: shape {C.shape}, type {type(C)}')

    _print_time = bool(os.environ.get('AUTOLIRPA_PRINT_TIME', 0))  # Check if timing should be printed

    if isinstance(C, str):
        # Handle batched CROWN if C is specified as a string
        if need_A_only or average_A:
            raise ValueError(
                'Batched CROWN is not compatible with '
                f'need_A_only={need_A_only}, average_A={average_A}'
            )
        ret = self.batched_backward(
            bound_node, C, unstable_idx,
            batch_size=roots[0].value.shape[0],
            bound_lower=bound_lower, bound_upper=bound_upper,
        )
        bound_node.lower, bound_node.upper = ret[:2]  # Assign the computed bounds
        return ret

    # Reset A matrices for all nodes
    for n in self.nodes():
        n.lA = n.uA = None

    degree_out = get_degrees(start_backpropagation_at_node)  # Get degrees of nodes for backpropagation
    C, batch_size, output_dim, output_shape = self._preprocess_C(C, bound_node)  # Preprocess C matrix

    if initial_As is None:
        # Initialize A matrices at the starting node
        start_backpropagation_at_node.lA = C if bound_lower else None
        start_backpropagation_at_node.uA = C if bound_upper else None
    else:
        # Assign initial A matrices if provided
        for layer_name, (lA, uA) in initial_As.items():
            self[layer_name].lA = lA
            self[layer_name].uA = uA
        assert start_backpropagation_at_node.lA is not None or start_backpropagation_at_node.uA is not None

    # Initialize lower and upper bounds
    if initial_lb is None:
        lb = torch.tensor(0., device=self.device)
    else:
        lb = initial_lb
    if initial_ub is None:
        ub = torch.tensor(0., device=self.device)
    else:
        ub = initial_ub

    A_record = {}  # Dictionary to record A matrices

    # Initialize a queue for breadth-first search (BFS) backpropagation
    queue = deque([start_backpropagation_at_node])
    while len(queue) > 0:
        l = queue.popleft()  # Dequeue the next node for backpropagation
        self.backward_from[l.name].append(bound_node)  # Track backpropagation origins

        if l.name in self.root_names:
            continue  # Skip root nodes

        # Enqueue predecessor nodes if all their outputs have been processed
        for l_pre in l.inputs:
            degree_out[l_pre.name] -= 1
            if degree_out[l_pre.name] == 0:
                queue.append(l_pre)

        # If the node has A matrices, proceed with backpropagation
        if l.lA is not None or l.uA is not None:
            if verbose:
                logger.debug(f'  Bound backward to {l} (out shape {l.output_shape})')
                if l.lA is not None:
                    logger.debug('    lA type %s shape %s',
                                 type(l.lA), list(l.lA.shape))
                if l.uA is not None:
                    logger.debug('    uA type %s shape %s',
                                 type(l.uA), list(l.uA.shape))

            if _print_time:
                start_time = time.time()

            if not l.perturbed:
                # If the node is not perturbed, add its bias directly to bounds
                if not hasattr(l, 'forward_value'):
                    self.get_forward_value(l)  # Ensure forward value is computed
                lb, ub = add_constant_node(lb, ub, l)
                continue

            if l.zero_uA_mtx and l.zero_lA_mtx:
                # If both A matrices are zero, no need to propagate
                continue

            lA, uA = l.lA, l.uA  # Retrieve A matrices
            if (l.name != start_backpropagation_at_node.name and use_beta_crown
                    and getattr(l, 'sparse_betas', None)):
                # Apply beta-CROWN for optimized bound computations
                lA, uA, lbias, ubias = self.beta_crown_backward_bound(
                    l, lA, uA, start_node=start_backpropagation_at_node)
                lb = lb + lbias  # Update lower bound
                ub = ub + ubias  # Update upper bound

            if isinstance(l, BoundOptimizableActivation):
                # Handle optimizable activation nodes
                if bound_node.name != self.final_node_name:
                    start_shape = bound_node.output_shape[1:]
                else:
                    start_shape = C.shape[0]
                l.preserve_mask = update_mask  # Preserve mask for activation
            else:
                start_shape = None

            # Perform bound backward computation for the node
            A, lower_b, upper_b = l.bound_backward(
                lA, uA, *l.inputs,
                start_node=bound_node, unstable_idx=unstable_idx,
                start_shape=start_shape
            )

            # Delete A matrices after use to save memory
            if bound_node.name != self.final_name:
                del l.lA, l.uA

            if _print_time:
                torch.cuda.synchronize()
                time_elapsed = time.time() - start_time
                if time_elapsed > 5e-3:
                    print(l, time_elapsed)

            if lb.ndim > 0 and isinstance(lower_b, Tensor) and self.conv_mode == 'patches':
                lb, ub, lower_b, upper_b = check_patch_biases(lb, ub, lower_b, upper_b)

            lb = lb + lower_b  # Update lower bound
            ub = ub + upper_b  # Update upper bound

            if self.return_A and self.needed_A_dict and bound_node.name in self.needed_A_dict:
                # If A matrices are needed, record them
                if len(self.needed_A_dict[bound_node.name]) == 0 or l.name in self.needed_A_dict[bound_node.name]:
                    # Handle A matrices based on their type
                    if isinstance(A[0][0], Patches):
                        A_record[l.name] = {
                            "lA": A[0][0],
                            "uA": A[0][1],
                            "lbias": lb.transpose(0, 1).detach() if lb.ndim > 1 else None,
                            "ubias": ub.transpose(0, 1).detach() if ub.ndim > 1 else None,
                            "unstable_idx": unstable_idx
                        }
                    else:
                        A_record[l.name] = {
                            "lA": A[0][0].transpose(0, 1).detach() if A[0][0] is not None else None,
                            "uA": A[0][1].transpose(0, 1).detach() if A[0][1] is not None else None,
                            "lbias": lb.transpose(0, 1).detach() if lb.ndim > 1 else None,
                            "ubias": ub.transpose(0, 1).detach() if ub.ndim > 1 else None,
                            "unstable_idx": unstable_idx
                        }

                # Merge the recorded A matrices into the overall A_dict
                self.A_dict.update({bound_node.name: A_record})

                if need_A_only and set(self.needed_A_dict[bound_node.name]) == set(A_record.keys()):
                    # If only A matrices are needed and all required A matrices are collected
                    self.A_dict.update({bound_node.name: A_record})
                    # Return early without computing bounds
                    return None, None, self.A_dict

            # Propagate the A matrices to predecessor nodes
            for i, l_pre in enumerate(l.inputs):
                add_bound(l, l_pre, lA=A[i][0], uA=A[i][1])

    # Transpose bounds if necessary
    if lb.ndim >= 2:
        lb = lb.transpose(0, 1)
    if ub.ndim >= 2:
        ub = ub.transpose(0, 1)

    if self.return_A and self.needed_A_dict and bound_node.name in self.needed_A_dict:
        # Save the A records if A matrices are returned
        save_A_record(
            bound_node, A_record, self.A_dict, roots,
            self.needed_A_dict[bound_node.name],
            lb=lb, ub=ub, unstable_idx=unstable_idx
        )

    # Handle cut constraints if used
    if (self.cut_used and getattr(self, 'cut_module', None) is not None
            and self.cut_module.x_coeffs is not None):
        # Propagate input neuron in cut constraints
        roots[0].lA, roots[0].uA = self.cut_module.input_cut(
            bound_node, roots[0].lA, roots[0].uA, roots[0].lower.size()[1:], unstable_idx,
            batch_mask=update_mask
        )

    # Concretize the bounds based on the accumulated lower and upper bounds
    lb, ub = concretize(self, batch_size, output_dim, lb, ub,
                        bound_lower, bound_upper,
                        average_A=average_A, node_start=bound_node)

    if tighten_input_bounds and isinstance(bound_node, BoundInput):
        # Tighten the input bounds if required
        shape = bound_node.perturbation.x_L.shape
        lb_reshaped = lb.reshape(shape)
        bound_node.perturbation.x_L = lb_reshaped - lb_reshaped.detach() + torch.max(bound_node.perturbation.x_L.detach(), lb_reshaped.detach())
        ub_reshaped = ub.reshape(shape)
        bound_node.perturbation.x_U = ub_reshaped - ub_reshaped.detach() + torch.min(bound_node.perturbation.x_U.detach(), ub_reshaped.detach())

    # Handle additional cut constraints if used
    if (self.cut_used and getattr(self, "cut_module", None) is not None
            and self.cut_module.cut_bias is not None):
        # Propagate cut bias in cut constraints
        lb, ub = self.cut_module.bias_cut(bound_node, lb, ub, unstable_idx, batch_mask=update_mask)
        if lb is not None and ub is not None and ((lb - ub) > 0).sum().item() > 0:
            # Warn if lower bound exceeds upper bound, indicating an error
            print(f"Warning: lb is larger than ub with diff: {(lb - ub)[(lb - ub) > 0].max().item()}")

    # Reshape the bounds to match the output shape
    lb = lb.view(batch_size, *output_shape) if bound_lower else None
    ub = ub.view(batch_size, *output_shape) if bound_upper else None

    if verbose:
        logger.debug('')  # Add an empty debug message for separation

    if torch.any(self.infeasible_bounds):
        # Adjust bounds for infeasible cases where lower bound exceeds upper bound
        if lb is not None:
            assert lb.size(0) == self.infeasible_bounds.size(0)
            lb = torch.where(self.infeasible_bounds.unsqueeze(1), torch.tensor(float('inf'), device=lb.device), lb)
        if ub is not None:
            assert ub.size(0) == self.infeasible_bounds.size(0)
            ub = torch.where(self.infeasible_bounds.unsqueeze(1), torch.tensor(float('-inf'), device=ub.device), ub)

    # Return the computed bounds along with A dictionaries if required
    if self.return_A:
        return lb, ub, self.A_dict
    else:
        return lb, ub


def get_unstable_size(unstable_idx):
    """
    Calculate the number of unstable neurons based on the unstable indices.

    Args:
        unstable_idx (torch.Tensor or tuple): Indices of unstable neurons.

    Returns:
        int: Number of unstable neurons.
    """
    if isinstance(unstable_idx, tuple):
        return unstable_idx[0].numel()
    else:
        return unstable_idx.numel()


def check_optimized_variable_sparsity(self: 'BoundedModule', node):
    """
    Check the sparsity of optimized variables (alpha values) for a given node.

    Args:
        self (BoundedModule): The bounded module instance.
        node (Bound): The node to check for sparsity.

    Returns:
        bool or None: True if alpha is sparse, False if dense, None if unknown.
    """
    alpha_sparsity = None  # Initialize as unknown

    for relu in self.relus:
        # Iterate through all ReLU nodes
        if relu.alpha_lookup_idx is not None and node.name in relu.alpha_lookup_idx:
            if relu.alpha_lookup_idx[node.name] is not None:
                # Alpha is sparse for this node
                alpha_sparsity = True
            elif self.bound_opts['optimize_bound_args']['use_shared_alpha']:
                # Shared alpha is used, and sparsity is supported
                alpha_sparsity = True
            else:
                # Alpha is dense
                alpha_sparsity = False
            break  # Found the relevant ReLU node

    return alpha_sparsity


def get_sparse_C(self: 'BoundedModule', node, ref_intermediate):
    """
    Determine and construct a sparse specification matrix (C) based on reference bounds.

    Args:
        self (BoundedModule): The bounded module instance.
        node (Bound): The node for which C is being constructed.
        ref_intermediate (tuple): Reference lower and upper bounds for intermediate layers.

    Returns:
        tuple: Contains the new C matrix, a flag indicating reduced dimensions, indices of unstable neurons, and their count.
    """
    (
        sparse_intermediate_bounds,
        ref_intermediate_lb,
        ref_intermediate_ub
    ) = ref_intermediate

    sparse_conv_intermediate_bounds = self.bound_opts.get('sparse_conv_intermediate_bounds', False)
    minimum_sparsity = self.bound_opts.get('minimum_sparsity', 0.9)
    crown_batch_size = self.bound_opts.get('crown_batch_size', 1e9)
    dim = int(prod(node.output_shape[1:]))
    batch_size = self.batch_size

    reduced_dim = False  # Indicates if only a subset of neurons are being bounded
    unstable_idx = None
    unstable_size = np.inf
    newC = None

    alpha_is_sparse = self.check_optimized_variable_sparsity(node)  # Check alpha sparsity

    # Handle different node types and determine the appropriate C matrix
    if (isinstance(node, BoundLinear) or isinstance(node, BoundMatMul)) and int(
            os.environ.get('AUTOLIRPA_USE_FULL_C', 0)) == 0:
        if sparse_intermediate_bounds:
            # Determine unstable neurons based on reference bounds
            unstable_idx, unstable_size = self.get_unstable_locations(
                ref_intermediate_lb, ref_intermediate_ub
            )
            if unstable_size == 0:
                # No unstable neurons; set reduced_dim to True and skip bounds
                reduced_dim = True
                unstable_idx = []
            elif unstable_size > crown_batch_size:
                # Use batched CROWN for large numbers of unstable neurons
                newC = 'OneHot'
                reduced_dim = True
            elif ((0 < unstable_size <= minimum_sparsity * dim
                   and alpha_is_sparse is None) or alpha_is_sparse):
                # Use sparse C for manageable numbers of unstable neurons
                newC = OneHotC(
                    [batch_size, unstable_size, *node.output_shape[1:]],
                    self.device,
                    unstable_idx,
                    None
                )
                reduced_dim = True
            else:
                # Fallback to dense C
                unstable_idx = None
                del ref_intermediate_lb, ref_intermediate_ub
        if not reduced_dim:
            # Use dense C if not reduced
            if dim > crown_batch_size:
                newC = 'eye'
            else:
                newC = eyeC([batch_size, dim, *node.output_shape[1:]], self.device)
    elif node.patches_start and node.mode == "patches":
        if sparse_intermediate_bounds:
            # Determine unstable neurons for patch-based nodes
            unstable_idx, unstable_size = self.get_unstable_locations(
                ref_intermediate_lb, ref_intermediate_ub, conv=True
            )
            if unstable_size == 0:
                reduced_dim = True
                unstable_idx = []
            elif unstable_size > crown_batch_size:
                newC = 'Patches'
                reduced_dim = True
            elif (sparse_conv_intermediate_bounds
                  and unstable_size <= minimum_sparsity * dim
                  and alpha_is_sparse is None) or alpha_is_sparse:
                # Create a sparse Patches object for efficient bound computation
                newC = Patches(
                    shape=[unstable_size, batch_size, *node.output_shape[1:-2], 1, 1],
                    identity=1,
                    unstable_idx=unstable_idx,
                    output_shape=[batch_size, *node.output_shape[1:]]
                )
                reduced_dim = True
            else:
                # Fallback to dense C
                unstable_idx = None
                del ref_intermediate_lb, ref_intermediate_ub
        if not reduced_dim:
            # Use dense Patches if not reduced
            newC = Patches(
                None, 1, 0,
                [node.output_shape[1], batch_size, *node.output_shape[2:], *node.output_shape[1:-2], 1, 1],
                1,
                output_shape=[batch_size, *node.output_shape[1:]]
            )
    elif (isinstance(node, (BoundAdd, BoundSub)) and node.mode == "patches"
          and len(node.output_shape) >= 4):
        # Handle patch-based addition or subtraction nodes
        if sparse_intermediate_bounds:
            # Determine unstable neurons
            unstable_idx, unstable_size = self.get_unstable_locations(
                ref_intermediate_lb, ref_intermediate_ub, conv=True
            )
            if unstable_size == 0:
                reduced_dim = True
                unstable_idx = []
            elif (sparse_conv_intermediate_bounds
                  and unstable_size <= minimum_sparsity * dim
                  and alpha_is_sparse is None) or alpha_is_sparse:
                # Create a sparse Patches object with specific patch sizes
                num_channel = node.output_shape[-3]
                patches = (
                    torch.eye(num_channel, device=self.device,
                              dtype=list(self.parameters())[0].dtype)).view(
                                  num_channel, 1, 1, 1, num_channel, 1, 1
                              )
                patches = patches.expand(-1, batch_size, node.output_shape[-2],
                                         node.output_shape[-1], -1, 1, 1)
                patches = patches[unstable_idx[0], :,
                                  unstable_idx[1], unstable_idx[2]]
                patches = patches.expand(-1, batch_size, -1, -1, -1)
                newC = Patches(
                    patches, 1, 0, patches.shape, unstable_idx=unstable_idx,
                    output_shape=[batch_size, *node.output_shape[1:]]
                )
                reduced_dim = True
            else:
                # Fallback to dense C
                unstable_idx = None
                del ref_intermediate_lb, ref_intermediate_ub
        if not reduced_dim:
            # Use dense Patches for addition or subtraction nodes
            num_channel = node.output_shape[-3]
            patches = (
                torch.eye(num_channel, device=self.device,
                          dtype=list(self.parameters())[0].dtype)).view(
                              num_channel, 1, 1, 1, num_channel, 1, 1
                          )
            patches = patches.expand(-1, batch_size, node.output_shape[-2],
                                     node.output_shape[-1], -1, 1, 1)
            newC = Patches(patches, 1, 0, patches.shape, output_shape=[
                batch_size, *node.output_shape[1:]
            ])
    else:
        # Handle other node types
        if sparse_intermediate_bounds:
            # Determine unstable neurons
            unstable_idx, unstable_size = self.get_unstable_locations(
                ref_intermediate_lb, ref_intermediate_ub
            )
            if unstable_size == 0:
                reduced_dim = True
                unstable_idx = []
            elif unstable_size > crown_batch_size:
                newC = 'eye'
                reduced_dim = True
            elif (unstable_size <= minimum_sparsity * dim
                  and alpha_is_sparse is None) or alpha_is_sparse:
                # Create a dense specification matrix
                newC = torch.zeros([1, unstable_size, dim], device=self.device)
                newC[0, torch.arange(unstable_size), unstable_idx] = 1.0
                newC = newC.expand(batch_size, -1, -1).view(
                    batch_size, unstable_size, *node.output_shape[1:]
                )
                reduced_dim = True
            else:
                # Fallback to dense C
                unstable_idx = None
                del ref_intermediate_lb, ref_intermediate_ub
        if not reduced_dim:
            # Use dense identity matrix for other node types
            if dim > 1000:
                warnings.warn(
                    f"Creating an identity matrix with size {dim}x{dim} for node {node}. "
                    "This may indicate poor performance for bound computation. "
                    "If you see this message on a small network please submit "
                    "a bug report.", stacklevel=2
                )
            if dim > crown_batch_size:
                newC = 'eye'
            else:
                newC = torch.eye(dim, device=self.device).unsqueeze(0).expand(
                    batch_size, -1, -1
                ).view(batch_size, dim, *node.output_shape[1:])
    return newC, reduced_dim, unstable_idx, unstable_size


def restore_sparse_bounds(self: 'BoundedModule', node, unstable_idx,
                          unstable_size, ref_intermediate,
                          new_lower=None, new_upper=None):
    """
    Restore sparse bounds for a given node based on reference bounds.

    Args:
        self (BoundedModule): The bounded module instance.
        node (Bound): The node for which bounds are being restored.
        unstable_idx (torch.Tensor or tuple): Indices of unstable neurons.
        unstable_size (int): Number of unstable neurons.
        ref_intermediate (tuple): Reference lower and upper bounds.
        new_lower (torch.Tensor, optional): New lower bounds. Defaults to None.
        new_upper (torch.Tensor, optional): New upper bounds. Defaults to None.
    """
    ref_intermediate_lb, ref_intermediate_ub = ref_intermediate[1:]

    batch_size = self.batch_size
    if unstable_size == 0:
        # No unstable neurons; set bounds to reference bounds
        node.lower = ref_intermediate_lb.detach().clone()
        node.upper = ref_intermediate_ub.detach().clone()
    else:
        if new_lower is None:
            new_lower = node.lower
        if new_upper is None:
            new_upper = node.upper

        if isinstance(unstable_idx, tuple):
            # Handle convolutional layers with multiple indices
            lower = ref_intermediate_lb.detach().clone()
            upper = ref_intermediate_ub.detach().clone()
            if len(unstable_idx) == 3:
                # For 3D indices (e.g., C, H, W)
                lower[:, unstable_idx[0], unstable_idx[1], unstable_idx[2]] = new_lower
                upper[:, unstable_idx[0], unstable_idx[1], unstable_idx[2]] = new_upper
            elif len(unstable_idx) == 4:
                # For 4D indices
                lower[:, unstable_idx[0], unstable_idx[1], unstable_idx[2], unstable_idx[3]] = new_lower
                upper[:, unstable_idx[0], unstable_idx[1], unstable_idx[2], unstable_idx[3]] = new_upper
        else:
            # Handle other layers with 1D indices
            lower = ref_intermediate_lb.detach().clone().view(batch_size, -1)
            upper = ref_intermediate_ub.detach().clone().view(batch_size, -1)
            lower[:, unstable_idx] = new_lower.view(batch_size, -1)
            upper[:, unstable_idx] = new_upper.view(batch_size, -1)

        # Reshape the bounds to match the node's output shape
        node.lower = lower.view(batch_size, *node.output_shape[1:])
        node.upper = upper.view(batch_size, *node.output_shape[1:])


def get_degrees(node_start):
    """
    Calculate the degrees (number of outgoing edges) for nodes starting from a given node.

    Args:
        node_start (Bound): The node from which to start calculating degrees.

    Returns:
        dict: A dictionary mapping node names to their degree counts.
    """
    if not isinstance(node_start, list):
        node_start = [node_start]
    degrees = {}
    added = {}
    queue = deque()

    # Initialize the queue with the starting node(s)
    for node in node_start:
        queue.append(node)
        added[node.name] = True

    # Perform BFS to calculate degrees
    while len(queue) > 0:
        l = queue.popleft()
        for l_pre in l.inputs:
            degrees[l_pre.name] = degrees.get(l_pre.name, 0) + 1
            if not added.get(l_pre.name, False):
                queue.append(l_pre)
                added[l_pre.name] = True

    return degrees


def _preprocess_C(self: 'BoundedModule', C, node):
    """
    Preprocess the specification matrix C based on the node's output shape.

    Args:
        self (BoundedModule): The bounded module instance.
        C (str or torch.Tensor): Specification matrix or a string indicating the type.
        node (Bound): The node for which C is being preprocessed.

    Returns:
        tuple: Contains the processed C matrix, batch size, output dimension, and output shape.
    """
    if isinstance(C, Patches):
        if C.unstable_idx is None:
            # Determine the output dimension based on C's shape
            if len(C.shape) == 7:
                out_c, batch_size, out_h, out_w = C.shape[:4]
                output_dim = out_c * out_h * out_w
            else:
                out_dim, batch_size, out_c, out_h, out_w = C.shape[:5]
                output_dim = out_dim * out_c * out_h * out_w
        else:
            # When unstable_idx is provided
            output_dim, batch_size = C.shape[:2]
    else:
        # Handle other types of C
        batch_size, output_dim = C.shape[:2]

    # Reshape C based on its type and node's output shape
    if not isinstance(C, (eyeC, Patches, OneHotC)):
        C = C.transpose(0, 1).reshape(
            output_dim, batch_size, *node.output_shape[1:]
        )
    elif isinstance(C, eyeC):
        C = C._replace(shape=(C.shape[1], C.shape[0], *C.shape[2:]))
    elif isinstance(C, OneHotC):
        C = C._replace(
            shape=(C.shape[1], C.shape[0], *C.shape[2:]),
            index=C.index.transpose(0, -1),
            coeffs=None if C.coeffs is None else C.coeffs.transpose(0, -1)
        )

    if isinstance(C, Patches) and C.unstable_idx is not None:
        # Set the output shape for sparse Patches
        output_shape = [C.shape[0]]
    elif prod(node.output_shape[1:]) != output_dim and not isinstance(C, Patches):
        # For nodes with mismatched output dimensions, use a generic shape
        output_shape = [-1]
    else:
        # Use the node's actual output shape
        output_shape = node.output_shape[1:]

    return C, batch_size, output_dim, output_shape


def concretize(self, batch_size, output_dim, lb, ub=None,
               bound_lower=True, bound_upper=True,
               average_A=False, node_start=None):
    """
    Concretize the accumulated lower and upper bounds by propagating through input nodes.

    Args:
        self (BoundedModule): The bounded module instance.
        batch_size (int): Number of samples in the batch.
        output_dim (int): Dimension of the output.
        lb (torch.Tensor or None): Lower bound tensor.
        ub (torch.Tensor or None): Upper bound tensor.
        bound_lower (bool, optional): If True, process lower bounds. Defaults to True.
        bound_upper (bool, optional): If True, process upper bounds. Defaults to True.
        average_A (bool, optional): If True, average the A matrices. Defaults to False.
        node_start (Bound, optional): The node where backpropagation started.

    Returns:
        tuple: Updated lower and upper bounds after concretization.
    """
    roots = self.roots()

    for i in range(len(roots)):
        if roots[i].lA is None and roots[i].uA is None:
            continue  # Skip nodes without A matrices

        if average_A and isinstance(roots[i], BoundParams):
            # Average the A matrices for BoundParams nodes
            lA = roots[i].lA.mean(
                node_start.batch_dim + 1, keepdim=True
            ).expand(roots[i].lA.shape) if bound_lower else None
            uA = roots[i].uA.mean(
                node_start.batch_dim + 1, keepdim=True
            ).expand(roots[i].uA.shape) if bound_upper else None
        else:
            lA, uA = roots[i].lA, roots[i].uA  # Use existing A matrices

        # Reshape A matrices if not using special types
        if not isinstance(roots[i].lA, eyeC) and not isinstance(roots[i].lA, Patches):
            lA = roots[i].lA.reshape(output_dim, batch_size, -1).transpose(0, 1) if bound_lower else None
        if not isinstance(roots[i].uA, eyeC) and not isinstance(roots[i].uA, Patches):
            uA = roots[i].uA.reshape(output_dim, batch_size, -1).transpose(0, 1) if bound_upper else None

        if hasattr(roots[i], 'perturbation') and roots[i].perturbation is not None:
            # Handle perturbations by concretizing the bounds
            if isinstance(roots[i], BoundParams):
                # For parameter nodes, handle batch dimensions
                lb = lb + roots[i].perturbation.concretize(
                    roots[i].center.unsqueeze(0), lA,
                    sign=-1, aux=roots[i].aux) if bound_lower else None
                ub = ub + roots[i].perturbation.concretize(
                    roots[i].center.unsqueeze(0), uA,
                    sign=+1, aux=roots[i].aux) if bound_upper else None
            else:
                # For non-parameter nodes
                lb = lb + roots[i].perturbation.concretize(
                    roots[i].center, lA, sign=-1, aux=roots[i].aux) if bound_lower else None
                ub = ub + roots[i].perturbation.concretize(
                    roots[i].center, uA, sign=+1, aux=roots[i].aux) if bound_upper else None
        else:
            # Handle nodes without perturbations by adding constant biases
            fv = roots[i].forward_value
            if type(roots[i]) == BoundInput:
                # For input nodes with a batch dimension
                batch_size_ = batch_size
            else:
                # For parameter nodes without a batch dimension
                batch_size_ = 1

            def _add_constant(A, b):
                if isinstance(A, eyeC):
                    b = b + fv.view(batch_size_, -1)
                elif isinstance(A, Patches):
                    b = b + A.matmul(fv, input_shape=roots[0].center.shape)
                elif type(roots[i]) == BoundInput:
                    b = b + A.matmul(fv.view(batch_size_, -1, 1)).squeeze(-1)
                else:
                    b = b + A.matmul(fv.view(-1, 1)).squeeze(-1)
                return b

            lb = _add_constant(lA, lb) if bound_lower else None
            ub = _add_constant(uA, ub) if bound_upper else None

    return lb, ub


def addA(A1, A2):
    """
    Add two A matrices (linear coefficients). Supports both Tensor and Patches types.

    Args:
        A1 (Tensor or Patches): First A matrix.
        A2 (Tensor or Patches): Second A matrix.

    Returns:
        Tensor or Patches: The resulting A matrix after addition.

    Raises:
        NotImplementedError: If A1 and A2 are of incompatible types.
    """
    if type(A1) == type(A2):
        return A1 + A2
    elif type(A1) == Patches:
        return A1 + A2
    elif type(A2) == Patches:
        return A2 + A1
    else:
        raise NotImplementedError(f'Unsupported types for A1 ({type(A1)}) and A2 ({type(A2)})')


def add_bound(node, node_pre, lA=None, uA=None):
    """
    Propagate A matrices (lA and uA) to a preceding node during backpropagation.

    Args:
        node (Bound): The current node in backpropagation.
        node_pre (Bound): The preceding node to which A matrices are propagated.
        lA (Tensor or Patches, optional): Lower A matrix. Defaults to None.
        uA (Tensor or Patches, optional): Upper A matrix. Defaults to None.
    """
    if lA is not None:
        if node_pre.lA is None:
            # First A matrix added to the preceding node
            node_pre.zero_lA_mtx = node.zero_backward_coeffs_l
            node_pre.lA = lA
        else:
            # Merge with existing A matrix
            node_pre.zero_lA_mtx = node_pre.zero_lA_mtx and node.zero_backward_coeffs_l
            new_node_lA = addA(node_pre.lA, lA)
            node_pre.lA = new_node_lA

    if uA is not None:
        if node_pre.uA is None:
            # First A matrix added to the preceding node
            node_pre.zero_uA_mtx = node.zero_backward_coeffs_u
            node_pre.uA = uA
        else:
            # Merge with existing A matrix
            node_pre.zero_uA_mtx = node_pre.zero_uA_mtx and node.zero_backward_coeffs_u
            node_pre.uA = addA(node_pre.uA, uA)


def add_constant_node(lb, ub, node):
    """
    Add the constant bias from a node to the accumulated bounds.

    Args:
        lb (torch.Tensor): Current lower bound tensor.
        ub (torch.Tensor): Current upper bound tensor.
        node (Bound): The node providing the constant bias.

    Returns:
        tuple: Updated lower and upper bounds.
    """
    new_lb = node.get_bias(node.lA, node.forward_value)  # Compute lower bias
    new_ub = node.get_bias(node.uA, node.forward_value)  # Compute upper bias

    # Ensure the dimensions match by reshaping if necessary
    if isinstance(lb, Tensor) and isinstance(new_lb, Tensor) and lb.ndim > 0 and lb.ndim != new_lb.ndim:
        new_lb = new_lb.reshape(lb.shape)
    if isinstance(ub, Tensor) and isinstance(new_ub, Tensor) and ub.ndim > 0 and ub.ndim != new_ub.ndim:
        new_ub = new_ub.reshape(ub.shape)

    lb = lb + new_lb  # Update lower bound
    ub = ub + new_ub  # Update upper bound
    return lb, ub


def save_A_record(node, A_record, A_dict, roots, needed_A_dict, lb, ub, unstable_idx):
    """
    Save the A matrices record for a specific node.

    Args:
        node (Bound): The node for which A matrices are being saved.
        A_record (dict): The current A matrices record.
        A_dict (dict): The overall A dictionary to be updated.
        roots (list): List of root nodes.
        needed_A_dict (dict): Dictionary specifying which A matrices are needed.
        lb (torch.Tensor): Lower bound tensor.
        ub (torch.Tensor): Upper bound tensor.
        unstable_idx (torch.Tensor or tuple): Indices of unstable neurons.
    """
    root_A_record = {}
    for i in range(len(roots)):
        if roots[i].lA is None and roots[i].uA is None:
            continue  # Skip nodes without A matrices

        if roots[i].name in needed_A_dict:
            # Process A matrices for required nodes
            if roots[i].lA is not None:
                if isinstance(roots[i].lA, Patches):
                    _lA = roots[i].lA
                else:
                    _lA = roots[i].lA.transpose(0, 1).detach()
            else:
                _lA = None

            if roots[i].uA is not None:
                if isinstance(roots[i].uA, Patches):
                    _uA = roots[i].uA
                else:
                    _uA = roots[i].uA.transpose(0, 1).detach()
            else:
                _uA = None

            # Record the A matrices and biases
            root_A_record.update({
                roots[i].name: {
                    "lA": _lA,
                    "uA": _uA,
                    "lbias": lb.detach() if lb.ndim > 1 else None,
                    "ubias": ub.detach() if ub.ndim > 1 else None,
                    "unstable_idx": unstable_idx
                }
            })
    root_A_record.update(A_record)  # Merge with existing A_record
    A_dict.update({node.name: root_A_record})  # Update the overall A_dict


def select_unstable_idx(ref_intermediate_lb, ref_intermediate_ub, unstable_locs, max_crown_size):
    """
    Select a subset of unstable neurons based on the loosest reference bounds when the number is too large.

    Args:
        ref_intermediate_lb (torch.Tensor): Reference lower bounds.
        ref_intermediate_ub (torch.Tensor): Reference upper bounds.
        unstable_locs (torch.Tensor): Boolean tensor indicating unstable locations.
        max_crown_size (int): Maximum number of neurons to process in CROWN.

    Returns:
        torch.Tensor: Selected indices of unstable neurons.
    """
    gap = (
        ref_intermediate_ub[:, unstable_locs]
        - ref_intermediate_lb[:, unstable_locs]
    ).sum(dim=0)  # Calculate the gap between upper and lower bounds

    indices = torch.argsort(gap, descending=True)  # Sort by descending gap
    indices_selected = indices[:max_crown_size]  # Select top gaps
    indices_selected, _ = torch.sort(indices_selected)  # Sort selected indices

    print(f'{len(indices_selected)}/{len(indices)} unstable neurons selected for CROWN')
    return indices_selected


def get_unstable_locations(self: 'BoundedModule', ref_intermediate_lb,
                           ref_intermediate_ub, conv=False, channel_only=False):
    """
    Identify the locations of unstable neurons based on reference bounds.

    Args:
        self (BoundedModule): The bounded module instance.
        ref_intermediate_lb (torch.Tensor): Reference lower bounds.
        ref_intermediate_ub (torch.Tensor): Reference upper bounds.
        conv (bool, optional): If True, handle convolutional layers. Defaults to False.
        channel_only (bool, optional): If True, only consider channel dimensions. Defaults to False.

    Returns:
        tuple: Contains the indices of unstable neurons and their count.
    """
    max_crown_size = self.bound_opts.get('max_crown_size', int(1e9))  # Maximum CROWN size

    # Identify neurons where lower bound < 0 and upper bound > 0 (unstable)
    unstable_masks = torch.logical_and(ref_intermediate_lb < 0, ref_intermediate_ub > 0)

    if channel_only:
        # Consider only channels with any unstable neurons
        unstable_locs = unstable_masks.sum(dim=(0, 2, 3)).bool()
        unstable_idx = unstable_locs.nonzero().squeeze(1)
    else:
        if not conv and unstable_masks.ndim > 2:
            # Flatten convolutional layers for non-conv nodes
            unstable_masks = unstable_masks.reshape(unstable_masks.size(0), -1)
            ref_intermediate_lb = ref_intermediate_lb.reshape(ref_intermediate_lb.size(0), -1)
            ref_intermediate_ub = ref_intermediate_ub.reshape(ref_intermediate_ub.size(0), -1)
        # Identify unstable locations
        unstable_locs = unstable_masks.sum(dim=0).bool()
        if conv:
            # For convolutional layers, get multi-dimensional indices
            unstable_idx = unstable_locs.nonzero(as_tuple=True)
        else:
            # For other layers, get linear indices
            unstable_idx = unstable_locs.nonzero().squeeze(1)

    unstable_size = get_unstable_size(unstable_idx)  # Number of unstable neurons

    if unstable_size > max_crown_size:
        # If too many unstable neurons, select a subset based on gaps
        indices_selected = select_unstable_idx(
            ref_intermediate_lb, ref_intermediate_ub, unstable_locs, max_crown_size
        )
        if isinstance(unstable_idx, tuple):
            # Update the indices for multi-dimensional tensors
            unstable_idx = tuple(u[indices_selected] for u in unstable_idx)
        else:
            unstable_idx = unstable_idx[indices_selected]

    unstable_size = get_unstable_size(unstable_idx)  # Update the count after selection

    return unstable_idx, unstable_size


def get_alpha_crown_start_nodes(
        self: 'BoundedModule',
        node,
        c=None,
        share_alphas=False,
        final_node_name=None,
    ):
    """
    Identify and return a list of start nodes for alpha-CROWN backpropagation.

    Args:
        self (BoundedModule): The bounded module instance.
        node (Bound): The current node.
        c (torch.Tensor, optional): Specification matrix. Defaults to None.
        share_alphas (bool, optional): If True, share alphas across layers. Defaults to False.
        final_node_name (str, optional): Name of the final node. Defaults to None.

    Returns:
        list: A list of tuples containing (node_name, node_shape, unstable_idx, is_final).
    """
    sparse_intermediate_bounds = self.bound_opts.get('sparse_intermediate_bounds', False)
    use_full_conv_alpha_thresh = self.bound_opts.get('use_full_conv_alpha_thresh', 512)

    start_nodes = []

    for nj in self.backward_from[node.name]:  # Iterate through nodes for backpropagation
        unstable_idx = None
        use_sparse_conv = None  # Indicates if sparse alpha is used for convolutional nodes
        use_full_conv_alpha = self.bound_opts.get('use_full_conv_alpha', False)

        # Determine unstable indices and sparsity for ReLU nodes
        if (sparse_intermediate_bounds
                and isinstance(node, BoundOptimizableActivation)
                and nj.name != final_node_name and not share_alphas):
            # Only handle ReLU activation nodes with intermediate bounds
            if len(nj.output_name) == 1 and isinstance(self[nj.output_name[0]], (BoundRelu, BoundSignMerge, BoundMaxPool)):
                if ((isinstance(nj, (BoundLinear, BoundMatMul)))
                        and int(os.environ.get('AUTOLIRPA_USE_FULL_C', 0)) == 0):
                    # Handle linear layers
                    unstable_idx, _ = self.get_unstable_locations(nj.lower, nj.upper)
                elif isinstance(nj, (BoundConv, BoundAdd, BoundSub, BoundBatchNormalization)) and nj.mode == 'patches':
                    if nj.name in node.patch_size:
                        # Handle patch-based convolutional layers
                        unstable_idx, _ = self.get_unstable_locations(
                            nj.lower, nj.upper, channel_only=not use_full_conv_alpha, conv=True
                        )
                        use_sparse_conv = False  # Shared alpha among channels
                        if use_full_conv_alpha and unstable_idx[0].size(0) > use_full_conv_alpha_thresh:
                            # Switch to shared alpha if too many unstable neurons
                            unstable_idx, _ = self.get_unstable_locations(
                                nj.lower, nj.upper, channel_only=True, conv=True
                            )
                            use_full_conv_alpha = False
                    else:
                        # Handle patch-converted convolutional layers
                        unstable_idx, _ = self.get_unstable_locations(nj.lower, nj.upper)
                        use_sparse_conv = True  # Sparse alpha in spec dimension
            else:
                # Handle other node types without specific patch sizes
                if isinstance(nj, (BoundConv, BoundAdd, BoundSub, BoundBatchNormalization)) and nj.mode == 'patches':
                    use_sparse_conv = False  # Sparse-spec alpha not used
        else:
            # Handle nodes without sparse intermediate bounds
            if isinstance(nj, (BoundConv, BoundAdd, BoundSub, BoundBatchNormalization)) and nj.mode == 'patches':
                use_sparse_conv = False  # Sparse-spec alpha not used

        if nj.name == final_node_name:
            # Always include the final node in the start nodes
            size_final = self[final_node_name].output_shape[-1] if c is None else c.size(1)
            start_nodes.append((final_node_name, size_final, None, True))
            continue

        if share_alphas:
            # Share alphas across all intermediate neurons
            output_shape = 1
        elif isinstance(node, BoundOptimizableActivation) and node.patch_size and nj.name in node.patch_size:
            # Handle patch-based activations
            if use_full_conv_alpha:
                # Alphas are not shared among channels
                output_shape = node.patch_size[nj.name][0], node.patch_size[nj.name][2], node.patch_size[nj.name][3]
            else:
                # Alphas are shared among channels
                output_shape = node.patch_size[nj.name][0]
            assert not sparse_intermediate_bounds or use_sparse_conv is False, "Alpha sparsity and convolutional alpha sharing conflict."
        else:
            # Use the node's output shape for non-patch-based activations
            assert not sparse_intermediate_bounds or use_sparse_conv is not False, "Alpha sparsity and convolutional alpha sharing conflict."
            output_shape = nj.lower.shape[1:]

        # Append the start node information
        start_nodes.append((nj.name, output_shape, unstable_idx, False))

    return start_nodes


def merge_A(batch_A, ret_A):
    """
    Merge two dictionaries of A matrices.

    Args:
        batch_A (dict): A dictionary of A matrices from the current batch.
        ret_A (dict): The cumulative dictionary of A matrices.

    Returns:
        dict: The merged dictionary of A matrices.
    """
    for key0 in batch_A:
        if key0 not in ret_A:
            ret_A[key0] = {}
        for key1 in batch_A[key0]:
            value = batch_A[key0][key1]
            if key1 not in ret_A[key0]:
                # Initialize the entry if it doesn't exist
                ret_A[key0].update({
                    key1: {
                        "lA": value["lA"],
                        "uA": value["uA"],
                        "lbias": value["lbias"],
                        "ubias": value["ubias"],
                    }
                })
            elif key0 == node.name:
                # Merge the A matrices for the current node
                exist = ret_A[key0][key1]

                if exist["unstable_idx"] is not None:
                    if isinstance(exist["unstable_idx"], torch.Tensor):
                        # Concatenate tensor indices
                        merged_unstable = torch.cat([
                            exist["unstable_idx"],
                            value['unstable_idx']
                        ], dim=0)
                    elif isinstance(exist["unstable_idx"], tuple):
                        if exist["unstable_idx"]:
                            # Concatenate tuple indices
                            merged_unstable = tuple([
                                torch.cat([exist["unstable_idx"][idx],
                                           value['unstable_idx'][idx]], dim=0)
                                for idx in range(len(exist['unstable_idx']))
                            ])
                        else:
                            merged_unstable = None
                    else:
                        raise NotImplementedError(
                            f'Unsupported type {type(exist["unstable_idx"])}')
                else:
                    merged_unstable = None

                # Merge lower and upper A matrices
                merge_dict = {"unstable_idx": merged_unstable}
                for name in ["lA", "uA"]:
                    if exist[name] is not None:
                        if isinstance(exist[name], torch.Tensor):
                            # Concatenate tensor A matrices
                            merge_dict[name] = torch.cat([exist[name], value[name]], dim=1)
                        else:
                            assert isinstance(exist[name], Patches)
                            # Concatenate Patches A matrices
                            merge_dict[name] = exist[name].create_similar(
                                torch.cat([exist[name].patches, value[name].patches], dim=0),
                                unstable_idx=merged_unstable
                            )
                    else:
                        merge_dict[name] = None

                # Merge biases
                for name in ["lbias", "ubias"]:
                    if exist[name] is not None:
                        # Concatenate bias tensors
                        merge_dict[name] = torch.cat([exist[name], value[name]], dim=1)
                    else:
                        merge_dict[name] = None

                # Update the merged A matrix
                ret_A[key0][key1] = merge_dict
    return ret_A


def concretize(self, batch_size, output_dim, lb, ub=None,
               bound_lower=True, bound_upper=True,
               average_A=False, node_start=None):
    """
    Concretize the accumulated bounds by propagating through input nodes and applying perturbations.

    Args:
        self (BoundedModule): The bounded module instance.
        batch_size (int): Number of samples in the batch.
        output_dim (int): Dimension of the output.
        lb (torch.Tensor): Lower bound tensor.
        ub (torch.Tensor): Upper bound tensor.
        bound_lower (bool, optional): If True, process lower bounds. Defaults to True.
        bound_upper (bool, optional): If True, process upper bounds. Defaults to True.
        average_A (bool, optional): If True, average the A matrices. Defaults to False.
        node_start (Bound, optional): The node where backpropagation started.

    Returns:
        tuple: Updated lower and upper bounds after concretization.
    """
    roots = self.roots()

    for i in range(len(roots)):
        if roots[i].lA is None and roots[i].uA is None:
            continue  # Skip nodes without A matrices

        if average_A and isinstance(roots[i], BoundParams):
            # Average the A matrices for BoundParams nodes
            lA = roots[i].lA.mean(
                node_start.batch_dim + 1, keepdim=True
            ).expand(roots[i].lA.shape) if bound_lower else None
            uA = roots[i].uA.mean(
                node_start.batch_dim + 1, keepdim=True
            ).expand(roots[i].uA.shape) if bound_upper else None
        else:
            lA, uA = roots[i].lA, roots[i].uA  # Use existing A matrices

        # Reshape A matrices if not using special types
        if not isinstance(roots[i].lA, eyeC) and not isinstance(roots[i].lA, Patches):
            lA = roots[i].lA.reshape(output_dim, batch_size, -1).transpose(0, 1) if bound_lower else None
        if not isinstance(roots[i].uA, eyeC) and not isinstance(roots[i].uA, Patches):
            uA = roots[i].uA.reshape(output_dim, batch_size, -1).transpose(0, 1) if bound_upper else None

        if hasattr(roots[i], 'perturbation') and roots[i].perturbation is not None:
            # Handle perturbations by concretizing the bounds
            if isinstance(roots[i], BoundParams):
                # For parameter nodes, handle batch dimensions
                lb = lb + roots[i].perturbation.concretize(
                    roots[i].center.unsqueeze(0), lA,
                    sign=-1, aux=roots[i].aux
                ) if bound_lower else None
                ub = ub + roots[i].perturbation.concretize(
                    roots[i].center.unsqueeze(0), uA,
                    sign=+1, aux=roots[i].aux
                ) if bound_upper else None
            else:
                # For non-parameter nodes
                lb = lb + roots[i].perturbation.concretize(
                    roots[i].center, lA, sign=-1, aux=roots[i].aux
                ) if bound_lower else None
                ub = ub + roots[i].perturbation.concretize(
                    roots[i].center, uA, sign=+1, aux=roots[i].aux
                ) if bound_upper else None
        else:
            # Handle nodes without perturbations by adding constant biases
            fv = roots[i].forward_value
            if type(roots[i]) == BoundInput:
                # For input nodes with a batch dimension
                batch_size_ = batch_size
            else:
                # For parameter nodes without a batch dimension
                batch_size_ = 1

            def _add_constant(A, b):
                if isinstance(A, eyeC):
                    b = b + fv.view(batch_size_, -1)
                elif isinstance(A, Patches):
                    b = b + A.matmul(fv, input_shape=roots[0].center.shape)
                elif type(roots[i]) == BoundInput:
                    b = b + A.matmul(fv.view(batch_size_, -1, 1)).squeeze(-1)
                else:
                    b = b + A.matmul(fv.view(-1, 1)).squeeze(-1)
                return b

            lb = _add_constant(lA, lb) if bound_lower else None
            ub = _add_constant(uA, ub) if bound_upper else None

    return lb, ub


def addA(A1, A2):
    """
    Add two A matrices (linear coefficients). Supports both Tensor and Patches types.

    Args:
        A1 (Tensor or Patches): First A matrix.
        A2 (Tensor or Patches): Second A matrix.

    Returns:
        Tensor or Patches: The resulting A matrix after addition.

    Raises:
        NotImplementedError: If A1 and A2 are of incompatible types.
    """
    if type(A1) == type(A2):
        return A1 + A2
    elif type(A1) == Patches:
        return A1 + A2
    elif type(A2) == Patches:
        return A2 + A1
    else:
        raise NotImplementedError(f'Unsupported types for A1 ({type(A1)}) and A2 ({type(A2)})')


def add_bound(node, node_pre, lA=None, uA=None):
    """
    Propagate A matrices (lA and uA) to a preceding node during backpropagation.

    Args:
        node (Bound): The current node in backpropagation.
        node_pre (Bound): The preceding node to which A matrices are propagated.
        lA (Tensor or Patches, optional): Lower A matrix. Defaults to None.
        uA (Tensor or Patches, optional): Upper A matrix. Defaults to None.
    """
    if lA is not None:
        if node_pre.lA is None:
            # First A matrix added to the preceding node
            node_pre.zero_lA_mtx = node.zero_backward_coeffs_l
            node_pre.lA = lA
        else:
            # Merge with existing A matrix
            node_pre.zero_lA_mtx = node_pre.zero_lA_mtx and node.zero_backward_coeffs_l
            new_node_lA = addA(node_pre.lA, lA)
            node_pre.lA = new_node_lA

    if uA is not None:
        if node_pre.uA is None:
            # First A matrix added to the preceding node
            node_pre.zero_uA_mtx = node.zero_backward_coeffs_u
            node_pre.uA = uA
        else:
            # Merge with existing A matrix
            node_pre.zero_uA_mtx = node_pre.zero_uA_mtx and node.zero_backward_coeffs_u
            node_pre.uA = addA(node_pre.uA, uA)


def add_constant_node(lb, ub, node):
    """
    Add the constant bias from a node to the accumulated bounds.

    Args:
        lb (torch.Tensor): Current lower bound tensor.
        ub (torch.Tensor): Current upper bound tensor.
        node (Bound): The node providing the constant bias.

    Returns:
        tuple: Updated lower and upper bounds.
    """
    new_lb = node.get_bias(node.lA, node.forward_value)  # Compute lower bias
    new_ub = node.get_bias(node.uA, node.forward_value)  # Compute upper bias

    # Ensure the dimensions match by reshaping if necessary
    if isinstance(lb, Tensor) and isinstance(new_lb, Tensor) and lb.ndim > 0 and lb.ndim != new_lb.ndim:
        new_lb = new_lb.reshape(lb.shape)
    if isinstance(ub, Tensor) and isinstance(new_ub, Tensor) and ub.ndim > 0 and ub.ndim != new_ub.ndim:
        new_ub = new_ub.reshape(ub.shape)

    lb = lb + new_lb  # Update lower bound
    ub = ub + new_ub  # Update upper bound
    return lb, ub


def save_A_record(node, A_record, A_dict, roots, needed_A_dict, lb, ub, unstable_idx):
    """
    Save the A matrices record for a specific node.

    Args:
        node (Bound): The node for which A matrices are being saved.
        A_record (dict): The current A matrices record.
        A_dict (dict): The overall A dictionary to be updated.
        roots (list): List of root nodes.
        needed_A_dict (dict): Dictionary specifying which A matrices are needed.
        lb (torch.Tensor): Lower bound tensor.
        ub (torch.Tensor): Upper bound tensor.
        unstable_idx (torch.Tensor or tuple): Indices of unstable neurons.
    """
    root_A_record = {}
    for i in range(len(roots)):
        if roots[i].lA is None and roots[i].uA is None:
            continue  # Skip nodes without A matrices

        if roots[i].name in needed_A_dict:
            # Process A matrices for required nodes
            if roots[i].lA is not None:
                if isinstance(roots[i].lA, Patches):
                    _lA = roots[i].lA
                else:
                    _lA = roots[i].lA.transpose(0, 1).detach()
            else:
                _lA = None

            if roots[i].uA is not None:
                if isinstance(roots[i].uA, Patches):
                    _uA = roots[i].uA
                else:
                    _uA = roots[i].uA.transpose(0, 1).detach()
            else:
                _uA = None

            # Record the A matrices and biases
            root_A_record.update({
                roots[i].name: {
                    "lA": _lA,
                    "uA": _uA,
                    "lbias": lb.detach() if lb.ndim > 1 else None,
                    "ubias": ub.detach() if ub.ndim > 1 else None,
                    "unstable_idx": unstable_idx
                }
            })
    root_A_record.update(A_record)  # Merge with existing A_record
    A_dict.update({node.name: root_A_record})  # Update the overall A_dict


def select_unstable_idx(ref_intermediate_lb, ref_intermediate_ub, unstable_locs, max_crown_size):
    """
    Select a subset of unstable neurons based on the loosest reference bounds when the number is too large.

    Args:
        ref_intermediate_lb (torch.Tensor): Reference lower bounds.
        ref_intermediate_ub (torch.Tensor): Reference upper bounds.
        unstable_locs (torch.Tensor): Boolean tensor indicating unstable locations.
        max_crown_size (int): Maximum number of neurons to process in CROWN.

    Returns:
        torch.Tensor: Selected indices of unstable neurons.
    """
    gap = (
        ref_intermediate_ub[:, unstable_locs]
        - ref_intermediate_lb[:, unstable_locs]
    ).sum(dim=0)  # Calculate the gap between upper and lower bounds

    indices = torch.argsort(gap, descending=True)  # Sort by descending gap
    indices_selected = indices[:max_crown_size]  # Select top gaps
    indices_selected, _ = torch.sort(indices_selected)  # Sort selected indices

    print(f'{len(indices_selected)}/{len(indices)} unstable neurons selected for CROWN')
    return indices_selected


def get_unstable_locations(self: 'BoundedModule', ref_intermediate_lb,
                           ref_intermediate_ub, conv=False, channel_only=False):
    """
    Identify the locations of unstable neurons based on reference bounds.

    Args:
        self (BoundedModule): The bounded module instance.
        ref_intermediate_lb (torch.Tensor): Reference lower bounds.
        ref_intermediate_ub (torch.Tensor): Reference upper bounds.
        conv (bool, optional): If True, handle convolutional layers. Defaults to False.
        channel_only (bool, optional): If True, only consider channel dimensions. Defaults to False.

    Returns:
        tuple: Contains the indices of unstable neurons and their count.
    """
    max_crown_size = self.bound_opts.get('max_crown_size', int(1e9))  # Maximum CROWN size

    # Identify neurons where lower bound < 0 and upper bound > 0 (unstable)
    unstable_masks = torch.logical_and(ref_intermediate_lb < 0, ref_intermediate_ub > 0)

    if channel_only:
        # Consider only channels with any unstable neurons
        unstable_locs = unstable_masks.sum(dim=(0, 2, 3)).bool()
        unstable_idx = unstable_locs.nonzero().squeeze(1)
    else:
        if not conv and unstable_masks.ndim > 2:
            # Flatten convolutional layers for non-conv nodes
            unstable_masks = unstable_masks.reshape(unstable_masks.size(0), -1)
            ref_intermediate_lb = ref_intermediate_lb.reshape(ref_intermediate_lb.size(0), -1)
            ref_intermediate_ub = ref_intermediate_ub.reshape(ref_intermediate_ub.size(0), -1)
        # Identify unstable locations
        unstable_locs = unstable_masks.sum(dim=0).bool()
        if conv:
            # For convolutional layers, get multi-dimensional indices
            unstable_idx = unstable_locs.nonzero(as_tuple=True)
        else:
            # For other layers, get linear indices
            unstable_idx = unstable_locs.nonzero().squeeze(1)

    unstable_size = get_unstable_size(unstable_idx)  # Number of unstable neurons

    if unstable_size > max_crown_size:
        # If too many unstable neurons, select a subset based on gaps
        indices_selected = select_unstable_idx(
            ref_intermediate_lb, ref_intermediate_ub, unstable_locs, max_crown_size
        )
        if isinstance(unstable_idx, tuple):
            # Update the indices for multi-dimensional tensors
            unstable_idx = tuple(u[indices_selected] for u in unstable_idx)
        else:
            unstable_idx = unstable_idx[indices_selected]

    unstable_size = get_unstable_size(unstable_idx)  # Update the count after selection

    return unstable_idx, unstable_size


def get_alpha_crown_start_nodes(
        self: 'BoundedModule',
        node,
        c=None,
        share_alphas=False,
        final_node_name=None,
    ):
    """
    Identify and return a list of start nodes for alpha-CROWN backpropagation.

    Args:
        self (BoundedModule): The bounded module instance.
        node (Bound): The current node.
        c (torch.Tensor, optional): Specification matrix. Defaults to None.
        share_alphas (bool, optional): If True, share alphas across layers. Defaults to False.
        final_node_name (str, optional): Name of the final node. Defaults to None.

    Returns:
        list: A list of tuples containing (node_name, node_shape, unstable_idx, is_final).
    """
    sparse_intermediate_bounds = self.bound_opts.get('sparse_intermediate_bounds', False)
    use_full_conv_alpha_thresh = self.bound_opts.get('use_full_conv_alpha_thresh', 512)

    start_nodes = []

    for nj in self.backward_from[node.name]:  # Iterate through nodes for backpropagation
        unstable_idx = None
        use_sparse_conv = None  # Indicates if sparse alpha is used for convolutional nodes
        use_full_conv_alpha = self.bound_opts.get('use_full_conv_alpha', False)

        # Determine unstable indices and sparsity for ReLU nodes
        if (sparse_intermediate_bounds
                and isinstance(node, BoundOptimizableActivation)
                and nj.name != final_node_name and not share_alphas):
            # Only handle ReLU activation nodes with intermediate bounds
            if len(nj.output_name) == 1 and isinstance(self[nj.output_name[0]], (BoundRelu, BoundSignMerge, BoundMaxPool)):
                if ((isinstance(nj, (BoundLinear, BoundMatMul)))
                        and int(os.environ.get('AUTOLIRPA_USE_FULL_C', 0)) == 0):
                    # Handle linear layers
                    unstable_idx, _ = self.get_unstable_locations(nj.lower, nj.upper)
                elif isinstance(nj, (BoundConv, BoundAdd, BoundSub, BoundBatchNormalization)) and nj.mode == 'patches':
                    if nj.name in node.patch_size:
                        # Handle patch-based convolutional layers
                        unstable_idx, _ = self.get_unstable_locations(
                            nj.lower, nj.upper, channel_only=not use_full_conv_alpha, conv=True
                        )
                        use_sparse_conv = False  # Shared alpha among channels
                        if use_full_conv_alpha and unstable_idx[0].size(0) > use_full_conv_alpha_thresh:
                            # Switch to shared alpha if too many unstable neurons
                            unstable_idx, _ = self.get_unstable_locations(
                                nj.lower, nj.upper, channel_only=True, conv=True
                            )
                            use_full_conv_alpha = False
                    else:
                        # Handle patch-converted convolutional layers
                        unstable_idx, _ = self.get_unstable_locations(nj.lower, nj.upper)
                        use_sparse_conv = True  # Sparse alpha in spec dimension
            else:
                # Handle other node types without specific patch sizes
                if isinstance(nj, (BoundConv, BoundAdd, BoundSub, BoundBatchNormalization)) and nj.mode == 'patches':
                    use_sparse_conv = False  # Sparse-spec alpha not used
        else:
            # Handle nodes without sparse intermediate bounds
            if isinstance(nj, (BoundConv, BoundAdd, BoundSub, BoundBatchNormalization)) and nj.mode == 'patches':
                use_sparse_conv = False  # Sparse-spec alpha not used

        if nj.name == final_node_name:
            # Always include the final node in the start nodes
            size_final = self[final_node_name].output_shape[-1] if c is None else c.size(1)
            start_nodes.append((final_node_name, size_final, None, True))
            continue

        if share_alphas:
            # Share alphas across all intermediate neurons
            output_shape = 1
        elif isinstance(node, BoundOptimizableActivation) and node.patch_size and nj.name in node.patch_size:
            # Handle patch-based activations
            if use_full_conv_alpha:
                # Alphas are not shared among channels
                output_shape = node.patch_size[nj.name][0], node.patch_size[nj.name][2], node.patch_size[nj.name][3]
            else:
                # Alphas are shared among channels
                output_shape = node.patch_size[nj.name][0]
            assert not sparse_intermediate_bounds or use_sparse_conv is False, "Alpha sparsity and convolutional alpha sharing conflict."
        else:
            # Use the node's output shape for non-patch-based activations
            assert not sparse_intermediate_bounds or use_sparse_conv is not False, "Alpha sparsity and convolutional alpha sharing conflict."
            output_shape = nj.lower.shape[1:]

        # Append the start node information
        start_nodes.append((nj.name, output_shape, unstable_idx, False))

    return start_nodes


def merge_A(batch_A, ret_A):
    """
    Merge two dictionaries of A matrices.

    Args:
        batch_A (dict): A dictionary of A matrices from the current batch.
        ret_A (dict): The cumulative dictionary of A matrices.

    Returns:
        dict: The merged dictionary of A matrices.
    """
    for key0 in batch_A:
        if key0 not in ret_A:
            ret_A[key0] = {}
        for key1 in batch_A[key0]:
            value = batch_A[key0][key1]
            if key1 not in ret_A[key0]:
                # Initialize the entry if it doesn't exist
                ret_A[key0].update({
                    key1: {
                        "lA": value["lA"],
                        "uA": value["uA"],
                        "lbias": value["lbias"],
                        "ubias": value["ubias"],
                    }
                })
            elif key0 == node.name:
                # Merge the A matrices for the current node
                exist = ret_A[key0][key1]

                if exist["unstable_idx"] is not None:
                    if isinstance(exist["unstable_idx"], torch.Tensor):
                        # Concatenate tensor indices
                        merged_unstable = torch.cat([
                            exist["unstable_idx"],
                            value['unstable_idx']
                        ], dim=0)
                    elif isinstance(exist["unstable_idx"], tuple):
                        if exist["unstable_idx"]:
                            # Concatenate tuple indices
                            merged_unstable = tuple([
                                torch.cat([exist["unstable_idx"][idx],
                                           value['unstable_idx'][idx]], dim=0)
                                for idx in range(len(exist['unstable_idx']))
                            ])
                        else:
                            merged_unstable = None
                    else:
                        raise NotImplementedError(
                            f'Unsupported type {type(exist["unstable_idx"])}')
                else:
                    merged_unstable = None

                # Merge lower and upper A matrices
                merge_dict = {"unstable_idx": merged_unstable}
                for name in ["lA", "uA"]:
                    if exist[name] is not None:
                        if isinstance(exist[name], torch.Tensor):
                            # Concatenate tensor A matrices
                            merge_dict[name] = torch.cat([exist[name], value[name]], dim=1)
                        else:
                            assert isinstance(exist[name], Patches)
                            # Concatenate Patches A matrices
                            merge_dict[name] = exist[name].create_similar(
                                torch.cat([exist[name].patches, value[name].patches], dim=0),
                                unstable_idx=merged_unstable
                            )
                    else:
                        merge_dict[name] = None

                # Merge biases
                for name in ["lbias", "ubias"]:
                    if exist[name] is not None:
                        # Concatenate bias tensors
                        merge_dict[name] = torch.cat([exist[name], value[name]], dim=1)
                    else:
                        merge_dict[name] = None

                # Update the merged A matrix
                ret_A[key0][key1] = merge_dict
    return ret_A
