/*
 * STRUMPACK -- STRUctured Matrices PACKage, Copyright (c) 2014, The
 * Regents of the University of California, through Lawrence Berkeley
 * National Laboratory (subject to receipt of any required approvals
 * from the U.S. Dept. of Energy).  All rights reserved.
 *
 * If you have questions about your rights to use or distribute this
 * software, please contact Berkeley Lab's Technology Transfer
 * Department at TTD@lbl.gov.
 *
 * NOTICE. This software is owned by the U.S. Department of Energy. As
 * such, the U.S. Government has been granted for itself and others
 * acting on its behalf a paid-up, nonexclusive, irrevocable,
 * worldwide license in the Software to reproduce, prepare derivative
 * works, and perform publicly and display publicly.  Beginning five
 * (5) years after the date permission to assert copyright is obtained
 * from the U.S. Department of Energy, and subject to any subsequent
 * five (5) year renewals, the U.S. Government is granted for itself
 * and others acting on its behalf a paid-up, nonexclusive,
 * irrevocable, worldwide license in the Software to reproduce,
 * prepare derivative works, distribute copies to the public, perform
 * publicly and display publicly, and to permit others to do so.
 *
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li.
 *             (Lawrence Berkeley National Lab, Computational Research
 *             Division).
 *
 */
#pragma once

#include <memory>
#include <vector>

namespace rxmesh {

/**
 * Helper class to construct a SeparatorTree.
 */
template <typename integer_t>
class Separator
{
   public:
    Separator(integer_t separator_end,
              integer_t parent,
              integer_t left_child,
              integer_t right_child)
        : sep_end(separator_end), pa(parent), lch(left_child), rch(right_child)
    {
    }
    integer_t sep_end, pa, lch, rch;
};


/**
 * Simple class to store a separator tree. A node in this tree
 * should always have 0 or 2 children.
 */
template <typename integer_t>
class SeparatorTree
{
   public:
    SeparatorTree() = default;
    SeparatorTree(integer_t nr_nodes);
    SeparatorTree(const std::vector<Separator<integer_t>>& seps);

    integer_t levels() const;
    integer_t level(integer_t i) const;
    integer_t root() const;
    void      print() const;
    void      printm(const std::string& name) const;
    void      check() const;

    SeparatorTree<integer_t> subtree(integer_t p, integer_t P) const;
    SeparatorTree<integer_t> toptree(integer_t P) const;

    integer_t separators() const
    {
        return nr_seps_;
    }

    bool is_leaf(integer_t sep) const
    {
        return lch[sep] == -1;
    }
    bool is_root(integer_t sep) const
    {
        return parent[sep] == -1;
    }
    bool is_empty() const
    {
        return nr_seps_ == 0;
    }

    integer_t *sizes = nullptr, *parent = nullptr, *lch = nullptr,
              *rch = nullptr;

   protected:
    integer_t              nr_seps_ = 0;
    std::vector<integer_t> iwork_;

    integer_t size() const
    {
        return 4 * nr_seps_ + 1;
    }

    void allocate(integer_t nseps)
    {
        nr_seps_ = nseps;
        iwork_.resize(4 * nseps + 1);
        sizes  = iwork_.data();
        parent = sizes + nr_seps_ + 1;
        lch    = parent + nr_seps_;
        rch    = lch + nr_seps_;
    }

   private:
    mutable integer_t root_ = -1;
};


/**
 * Create a separator tree based on a matrix and a
 * permutation. First build the elimination tree, then postorder the
 * elimination tree. Then combine the postordering of the
 * elimination tree with the permutation. Build a separator tree
 * from the elimination tree. Set the inverse permutation.
 */
template <typename integer_t>
SeparatorTree<integer_t> build_sep_tree_from_perm(
    const integer_t*        ptr,
    const integer_t*        ind,
    std::vector<integer_t>& perm,
    std::vector<integer_t>& iperm);

/*! \brief Symmetric elimination tree
 *
 * <pre>
 *      p = spsymetree (A);
 *
 *      Find the elimination tree for symmetric matrix A.
 *      This uses Liu's algorithm, and runs in time O(nz*log n).
 *
 *      Input:
 *        Square sparse matrix A.  No check is made for symmetry;
 *        elements below and on the diagonal are ignored.
 *        Numeric values are ignored, so any explicit zeros are
 *        treated as nonzero.
 *      Output:
 *        Integer array of parents representing the etree, with n
 *        meaning a root of the elimination forest.
 *      Note:
 *        This routine uses only the upper triangle, while sparse
 *        Cholesky (as in spchol.c) uses only the lower.  Matlab's
 *        dense Cholesky uses only the upper.  This routine could
 *        be modified to use the lower triangle either by transposing
 *        the matrix or by traversing it by rows with auxiliary
 *        pointer and link arrays.
 *
 *      John R. Gilbert, Xerox, 10 Dec 1990
 *      Based on code by JRG dated 1987, 1988, and 1990.
 *      Modified by X.S. Li, November 1999.
 * </pre>
 */
template <typename integer_t>
std::vector<integer_t> spsymetree(
    const integer_t* acolst,        // column starts
    const integer_t* acolend,       //   and ends past 1
    const integer_t* arow,          // row indices of A
    integer_t        n,             // dimension of A
    integer_t        subgraph_begin = 0);  // first row/column of subgraph

template <typename integer_t>
std::vector<integer_t> etree_postorder(const std::vector<integer_t>& etree);

template <typename integer_t>
std::vector<Separator<integer_t>> separators_from_etree(
    std::vector<integer_t>& etree,
    std::vector<integer_t>& post);

}  // namespace rxmesh

