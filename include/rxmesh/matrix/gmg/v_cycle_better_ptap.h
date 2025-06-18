#pragma once
#include "v_cycle.h"

using namespace rxmesh;

template <typename T>
struct VCycle_Better : public rxmesh::VCycle<T>
{
    using rxmesh::VCycle<T>::VCycle;  // Inherit constructors

    // Override the new_ptap method
    void new_ptap(SparseMatrix<T> p,
                  SparseMatrix<T> new_a,
                  SparseMatrix<T> old_a) override
    {
        // Your custom implementation here
        printf("Custom ptap called!\n");

        // Example: call base implementation if needed
        // rxmesh::VCycle<T>::new_ptap(p, new_a, old_a);
    }

    /*__host__ void new_ptap(SparseMatrix<T> p,
                           SparseMatrix<T> new_a,
                           SparseMatrix<T> old_a)
    {*/
        /*
        constexpr uint32_t blockThreads = 256;
        uint32_t           blocks_new   = DIVIDE_UP(new_a.rows(), blockThreads);
        uint32_t           blocks_old   = DIVIDE_UP(old_a.rows(), blockThreads);


        // first make sets per row
        for_each_item<<<blocks_old, blockThreads>>>(
            old_a.rows(), [old_a, p,match_store] __device__(int i) mutable {
                //if (i == 0) {
                    for (int q = old_a.row_ptr()[i]; q < old_a.row_ptr()[i + 1];
        ++q) { int a_col = old_a.col_id(i, q - old_a.row_ptr()[i]); int a_index
        = q - old_a.row_ptr()[i];
                        //printf("\n%d is connected to %d", i, a_col);

                        // go through each coarse vertex, and all the things
                        // they influence
                        for (int x = 0; x < p.rows(); x += 1)
                            for (int p_iter = p.row_ptr()[x];
                                 p_iter < p.row_ptr()[x + 1];
                                 ++p_iter)
                            {

                                int p_index = p_iter - p.row_ptr()[x];
                                int p_col = p.col_id(x,p_index);
                                /*printf(
                                    "\nComparing: p(%d, %d) = %d vs a(%d, %d) "
                                    "= "
                                    "%d",
                                    x,
                                    p_index,
                                    p_col,
                                    i,
                                    a_index,
                                    a_col);
                                if (p_col == a_col)
                                {
                                    /*printf(
                                        "\ncoarse vertex i=%d influences fine "
                                        "vertex j=%d hence (%d,%d) is non zero "
                                        "on PtA",
                                        x,
                                        a_col,
                                        x,
                                        i);
                                    Edge pair(x, i);
                                    match_store.insert(pair);  // Thread-safe
        atomic insert
                                }
                                break;
                            }
                    }
                //}

            });

        match_store.uniquify();

        match_store.for_each(
            [new_a,p] __device__(Edge e) mutable
        {
            std::pair<int, int> q = e.unpack();
            int                 a = q.first;
            int                 b = q.second;
            //printf("\n(%d,%d)", a, b);

            for (int x = 0; x < p.rows(); x += 1)
                for (int p_iter = p.row_ptr()[x]; p_iter < p.row_ptr()[x +
        1];++p_iter)
                {
                    int p_index = p_iter - p.row_ptr()[x];
                    int p_col   = p.col_id(x, p_index);
                    if (b==p_col) {
                        if (x==5)
                        printf("\n(%d,%d) is non zero in the next laplacian,
        sets were (%d,%d) and (%d,%d)", x, a,a,b,x,p_col);
                    }

                }
        });

         for_each_item<<<blocks_new, blockThreads>>>(
            n, [old_a, new_a, p,match_store] __device__(int i) mutable
         {

             //go through each coarse row

             //check if anything in the set matches with what is influencing



         });

        /*
        for_each_item<<<blocks, blockThreads>>>(n,
            [old_a,new_a,p] __device__(int i) mutable
            {
                //printf("\n%d is influenced by triangle made of %d %d
        %d",i,p.col_id(i,0),p.col_id(i,1),p.col_id(i,2));

            //printf("\n%d influences")

            //for an entry in new A to be non zero-> the
            if (i==0)
            for (int q = p.row_ptr()[i]; q < p.row_ptr()[i + 1]; q++) {
                    printf("\n%d is influenced by %d",
                       p.col_id(i, q - p.row_ptr()[i]),
                           i);

                    //in those influenced, gather the 1 ring

                }
                //use A for connectivity determinations
                for (int j = 0; j < new_a.rows();j++) {

                    //go through

                    //printf("\nPtA non zeros:");

                }
            });
            */
    //}




    void get_intermediate_laplacians(GMG<T>& gmg, SparseMatrix<T>& A) override
    {
        SparseMatrix<T> p_t = gmg.m_prolong_op[0].transpose();
        new_ptap(p_t,m_a[0].a, A);
    }
};
