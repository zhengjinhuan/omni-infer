// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include "omni_radix_tree.h"
#include <assert.h>
#include <stdlib.h>

static uint64_t omni_generate_node_id(uint64_t parent_hash, uint64_t child_hash)
{
    u_char key[16];

    *(uint64_t *)key = parent_hash;
    *(uint64_t *)(key + 8) = child_hash;

    return ngx_murmur_hash2(key, 16);
}

omni_radix_tree_t *omni_radix_tree_init(ngx_slab_pool_t *shpool)
{
    omni_radix_tree_t *tree;

    tree = ngx_slab_alloc(shpool, sizeof(omni_radix_tree_t));
    if (tree == NULL)
    {
        return NULL;
    }

    ngx_memzero(tree, sizeof(omni_radix_tree_t));
    tree->shpool = shpool;
    tree->version = 1;

    if (ngx_shmtx_create(&tree->mutex, &tree->lock, (u_char *)"radix_lock") != NGX_OK)
    {
        ngx_slab_free(shpool, tree);
        return NULL;
    }

    ngx_rbtree_init(&tree->lookup_tree, &tree->sentinel, ngx_rbtree_insert_value);

    tree->root = ngx_slab_alloc(shpool, sizeof(omni_radix_node_t));
    if (tree->root == NULL)
    {
        ngx_shmtx_destroy(&tree->mutex);
        ngx_slab_free(shpool, tree);
        return NULL;
    }

    ngx_memzero(tree->root, sizeof(omni_radix_node_t));
    tree->root->hash_key = 0;
    tree->root->rb_node.key = omni_generate_node_id(0, 0);
    tree->root->parent = NULL;
    tree->root->first_child = NULL;
    tree->root->next_sibling = NULL;

    ngx_rbtree_insert(&tree->lookup_tree, &tree->root->rb_node);

    tree->writer_pid = ngx_pid;

    return tree;
}

static omni_radix_node_t *omni_find_node(omni_radix_tree_t *tree, uint64_t parent_hash, uint64_t child_hash)
{
    ngx_rbtree_node_t *node;
    uint64_t node_id;

    node_id = omni_generate_node_id(parent_hash, child_hash);

    node = tree->lookup_tree.root;
    while (node != NULL && node != &tree->sentinel)
    {
        if (node_id < node->key)
        {
            node = node->left;
        }
        else if (node_id > node->key)
        {
            node = node->right;
        }
        else
        {
            return (omni_radix_node_t *)node;
        }
    }

    return NULL;
}

ngx_int_t omni_radix_tree_add_chain(omni_radix_tree_t *tree, uint64_t *hash_chain,
                                    ngx_uint_t chain_len)
{
    omni_radix_node_t *current, *new_node;
    ngx_uint_t i;

    if (chain_len == 0 || hash_chain == NULL)
    {
        return NGX_ERROR;
    }

    ngx_shmtx_lock(&tree->mutex);

    current = tree->root;

    for (i = 0; i < chain_len; i++)
    {
        uint64_t current_hash = hash_chain[i];
        omni_radix_node_t *existing;
        uint64_t node_id;

        existing = omni_find_node(tree, current->hash_key, current_hash);

        if (existing != NULL)
        {
            current = existing;
            continue;
        }

        new_node = ngx_slab_alloc(tree->shpool, sizeof(omni_radix_node_t));
        if (new_node == NULL)
        {
            ngx_shmtx_unlock(&tree->mutex);
            return NGX_ERROR;
        }

        ngx_memzero(new_node, sizeof(omni_radix_node_t));
        new_node->hash_key = current_hash;

        node_id = omni_generate_node_id(current->hash_key, current_hash);
        new_node->rb_node.key = node_id;

        /* link into sibling list and set parent pointer */
        new_node->next_sibling = current->first_child;
        new_node->first_child = NULL;
        new_node->parent = current;
        current->first_child = new_node;

        ngx_rbtree_insert(&tree->lookup_tree, &new_node->rb_node);

        current = new_node;
    }

    tree->version++;

    ngx_shmtx_unlock(&tree->mutex);
    return NGX_OK;
}

ngx_uint_t omni_radix_tree_match(omni_radix_tree_t *tree, uint64_t *hash_chain,
                                 ngx_uint_t chain_len)
{
    omni_radix_node_t *current;
    ngx_uint_t match_depth = 0;
    ngx_uint_t i;

    if (chain_len == 0 || hash_chain == NULL)
    {
        return 0;
    }

    ngx_shmtx_lock(&tree->mutex);

    current = tree->root;

    for (i = 0; i < chain_len; i++)
    {
        uint64_t current_hash = hash_chain[i];
        omni_radix_node_t *next;

        next = omni_find_node(tree, current->hash_key, current_hash);

        if (next == NULL)
        {
            break;
        }

        if (next != tree->root)
        {
            match_depth++;
        }

        current = next;
    }

    ngx_shmtx_unlock(&tree->mutex);
    return match_depth;
}

ngx_uint_t omni_radix_tree_match_optimistic(omni_radix_tree_t *tree, uint64_t *hash_chain,
                                            ngx_uint_t chain_len)
{
    omni_radix_node_t *current;
    ngx_uint_t match_depth = 0;
    ngx_uint_t i, start_version;

    if (chain_len == 0 || hash_chain == NULL)
    {
        return 0;
    }

    do
    {
        ngx_memory_barrier();
        start_version = tree->version;

        current = tree->root;
        match_depth = 0;

        for (i = 0; i < chain_len; i++)
        {
            uint64_t current_hash = hash_chain[i];
            omni_radix_node_t *next = NULL;
            uint64_t node_id;

            node_id = omni_generate_node_id(current->hash_key, current_hash);
            ngx_rbtree_node_t *node = tree->lookup_tree.root;

            while (node != &tree->sentinel)
            {
                if (node_id < node->key)
                {
                    node = node->left;
                }
                else if (node_id > node->key)
                {
                    node = node->right;
                }
                else
                {
                    next = (omni_radix_node_t *)node;
                    break;
                }
            }

            if (next == NULL)
            {
                break;
            }

            match_depth++;
            current = next;
        }

        ngx_memory_barrier();
    } while (tree->version != start_version);

    return match_depth;
}

static void omni_recursive_remove(omni_radix_tree_t *tree, omni_radix_node_t *node)
{
    omni_radix_node_t *child, *next_child;

    if (node == NULL)
    {
        return;
    }

    child = node->first_child;
    while (child != NULL)
    {
        next_child = child->next_sibling;
        omni_recursive_remove(tree, child);
        child = next_child;
    }



    ngx_rbtree_delete(&tree->lookup_tree, &node->rb_node);
    ngx_slab_free(tree->shpool, node);
}

static void omni_unlink_from_parent(omni_radix_tree_t *tree, omni_radix_node_t *node)
{
    if (node == NULL || node->parent == NULL)
        return;

    omni_radix_node_t *parent = node->parent;
    if (parent->first_child == node)
    {
        parent->first_child = node->next_sibling;
    }
    else
    {
        omni_radix_node_t *prev = parent->first_child;
        while (prev != NULL && prev->next_sibling != node)
        {
            prev = prev->next_sibling;
        }
        if (prev != NULL)
        {
            prev->next_sibling = node->next_sibling;
        }
    }
    node->parent = NULL;
    node->next_sibling = NULL;
}

static omni_radix_node_t *omni_find_node_by_hash(omni_radix_tree_t *tree, uint64_t target)
{
    ngx_rbtree_node_t *root = tree->lookup_tree.root;
    if (root == &tree->sentinel)
        return NULL;

    /* initial stack capacity */
    size_t stack_cap = 256;
    ngx_rbtree_node_t **stack = malloc(stack_cap * sizeof(ngx_rbtree_node_t *));
    if (!stack)
        return NULL;

    size_t sp = 0;
    stack[sp++] = root;

    omni_radix_node_t *found = NULL;

    while (sp > 0)
    {
        ngx_rbtree_node_t *n = stack[--sp];
        if (n == &tree->sentinel)
            continue;

        omni_radix_node_t *orn = (omni_radix_node_t *)n;
        if (orn != tree->root && orn->hash_key == target)
        {
            found = orn;
            break;
        }

        if (n->right != &tree->sentinel)
        {
            if (sp + 1 >= stack_cap)
            {
                size_t new_cap = stack_cap * 2;
                ngx_rbtree_node_t **new_stack = realloc(stack, new_cap * sizeof(ngx_rbtree_node_t *));
                if (!new_stack)
                {
                    break;
                }
                stack = new_stack;
                stack_cap = new_cap;
            }
            stack[sp++] = n->right;
        }
        if (n->left != &tree->sentinel)
        {
            if (sp + 1 >= stack_cap)
            {
                size_t new_cap = stack_cap * 2;
                ngx_rbtree_node_t **new_stack = realloc(stack, new_cap * sizeof(ngx_rbtree_node_t *));
                if (!new_stack)
                {
                    break;
                }
                stack = new_stack;
                stack_cap = new_cap;
            }
            stack[sp++] = n->left;
        }
    }

    free(stack);
    return found;
}

ngx_int_t omni_radix_tree_remove(omni_radix_tree_t *tree, uint64_t hash_to_remove)
{
    if (tree == NULL)
    {
        return NGX_ERROR;
    }

    ngx_shmtx_lock(&tree->mutex);

    ngx_int_t removed_any = NGX_ERROR;


    while (1)
    {
        omni_radix_node_t *node = omni_find_node_by_hash(tree, hash_to_remove);
        if (node == NULL)
            break;

        /* unlink from parent */
        omni_unlink_from_parent(tree, node);

        omni_recursive_remove(tree, node);

        removed_any = NGX_OK;
    }

    if (removed_any == NGX_OK)
    {
        tree->version++;
    }

    ngx_shmtx_unlock(&tree->mutex);
    return removed_any;
}

void omni_radix_tree_destroy(omni_radix_tree_t *tree)
{
    if (tree == NULL)
    {
        return;
    }

    ngx_shmtx_lock(&tree->mutex);

    if (tree->root)
    {
        omni_recursive_remove(tree, tree->root);
        tree->root = NULL;
    }

    ngx_shmtx_unlock(&tree->mutex);
    ngx_shmtx_destroy(&tree->mutex);

    ngx_slab_free(tree->shpool, tree);
}

/* --- tests retained --- */

static void omni_radix_tree_test_internal(omni_radix_tree_t *tree, ngx_uint_t (*matcher)(omni_radix_tree_t *, uint64_t *, ngx_uint_t))
{
    ngx_uint_t match_depth;

    printf("\n--- Running tests with omni_radix_tree_match ---\n");

    // Test 1: Basic chain addition and matching
    printf("Test 1: Basic chain operations\n");
    uint64_t chain1[] = {128803, 7282};
    assert(omni_radix_tree_add_chain(tree, chain1, 2) == NGX_OK);

    match_depth = matcher(tree, chain1, 2);
    assert(match_depth == 2);

    // Test 2: Partial matching
    printf("Test 2: Partial matching\n");
    uint64_t query1[] = {128803, 7282, 9999, 8888};
    match_depth = matcher(tree, query1, 4);
    assert(match_depth == 2);

    // Test 3: Add another chain with common prefix
    printf("Test 3: Chains with common prefix\n");
    uint64_t chain2[] = {128803, 2282};
    assert(omni_radix_tree_add_chain(tree, chain2, 2) == NGX_OK);

    match_depth = matcher(tree, chain2, 2);
    assert(match_depth == 2);

    // Test 4: Test common prefix matching
    printf("Test 4: Common prefix matching\n");
    uint64_t query2[] = {128803, 1182};
    match_depth = matcher(tree, query2, 2);
    assert(match_depth == 1);

    // Test 5: Add longer chain
    printf("Test 5: Longer chains\n");
    uint64_t chain3[] = {128803, 1182, 5567};
    assert(omni_radix_tree_add_chain(tree, chain3, 3) == NGX_OK);

    match_depth = matcher(tree, chain3, 3);
    assert(match_depth == 3);

    // Test 6: Test longest prefix matching
    printf("Test 6: Longest prefix matching\n");
    uint64_t query3[] = {128803, 1182, 5567, 2342, 34242};
    match_depth = matcher(tree, query3, 5);
    assert(match_depth == 3);

    // Test 7: Add chain with different starting point
    printf("Test 7: Different starting chains\n");
    uint64_t chain4[] = {24928, 128803, 7282};
    assert(omni_radix_tree_add_chain(tree, chain4, 3) == NGX_OK);

    match_depth = matcher(tree, chain4, 3);
    assert(match_depth == 3);

    // Test 8: Test non-matching chain
    printf("Test 8: Non-matching chains\n");
    uint64_t query4[] = {999999, 8888};
    match_depth = matcher(tree, query4, 2);
    assert(match_depth == 0);

    // Test 9: Test empty query
    printf("Test 9: Empty query\n");
    match_depth = matcher(tree, NULL, 0);
    assert(match_depth == 0);

    // Test 10: Remove operation
    printf("Test 10: Remove operation\n");
    assert(omni_radix_tree_remove(tree, 128803) == NGX_OK);

    // Verify removal worked -
    match_depth = matcher(tree, chain1, 2);
    assert(match_depth == 0);

    match_depth = matcher(tree, chain2, 2);
    assert(match_depth == 0);

    match_depth = matcher(tree, chain3, 3);
    assert(match_depth == 0);

    // Test 11: Try to remove non-existent node
    printf("Test 11: Remove non-existent node\n");
    assert(omni_radix_tree_remove(tree, 999999) == NGX_ERROR);

    // Test 12: Add chains after removal
    printf("Test 12: Add after removal\n");
    uint64_t chain5[] = {55555, 66666};
    assert(omni_radix_tree_add_chain(tree, chain5, 2) == NGX_OK);

    match_depth = matcher(tree, chain5, 2);
    assert(match_depth == 2);

    // Test 13: Edge case - single element chain
    printf("Test 13: Single element chains\n");
    uint64_t chain6[] = {77777};
    assert(omni_radix_tree_add_chain(tree, chain6, 1) == NGX_OK);

    match_depth = matcher(tree, chain6, 1);
    assert(match_depth == 1);

    // Test 14: Test exact matching after various operations
    printf("Test 14: Exact matching verification\n");
    uint64_t exact_chain[] = {55555, 66666};
    match_depth = matcher(tree, exact_chain, 2);
    assert(match_depth == 2);

    // Test 15: Test chain with zero as data value (not root)
    printf("Test 15: Chain with zero data value\n");
    uint64_t chain7[] = {123, 0, 456};
    assert(omni_radix_tree_add_chain(tree, chain7, 3) == NGX_OK);

    match_depth = matcher(tree, chain7, 3);
    assert(match_depth == 3);

    printf("All tests passed successfully!\n");
}

void omni_radix_tree_test(omni_radix_tree_t *tree)
{
    omni_radix_tree_test_internal(tree, omni_radix_tree_match);
    omni_radix_tree_test_internal(tree, omni_radix_tree_match_optimistic);
}