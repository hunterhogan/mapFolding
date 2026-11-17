// Statistics collector adapted from the attributed Sawada-Li generator below.
// Build: gcc -O3 -march=native -std=c11 research/permutationStatistics.c -o /tmp/permutationStatistics
// Run: /tmp/permutationStatistics 15
// Enumerates the A000682(n) permutations starting with 1.
// Run statistics use inverse-permutation direction changes and Sade first-run lengths,
// as defined by the PARI programs in the bundled A259698/A259701/A259703 entries.
// n<=32 keeps every histogram below Catalan(32), safely inside uint64_t.
//------------------------------------------------------
// STAMP FOLDINGS, SEMI-MEANDERS, OPEN MEANDERS
// Research by:   Joe Sawada, Roy Li
// Programmed by: Joe Sawada 2008
// Available at:  http://www.socs.uoguelph.ca/~sawada/programs.html
//------------------------------------------------------
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>
#define MAX(a,b) ((a > b) ? a : b)
#define MAX_VAL 99
#undef NULL
#define NULL -1

typedef struct List {
    int head, tail;
} List;

typedef struct Interval {
    int left, right, prev, next, node_ptr, next_perm, prev_perm;
} Interval;

typedef struct Node {
    List L, R;
    int up_interval;
    char up_side;
} Node;

//--------------------------------------------------------------------------------
// GLOBAL VARIABLES
//--------------------------------------------------------------------------------
Interval interval[MAX_VAL];
List perm;
Node node[MAX_VAL];
int N, MEANDER=0, SEMI=1, STAMP=0, UNLABELED=0, SYM_SEMI=0;
uint64_t total=0, runs_hist[MAX_VAL]={0}, first_hist[MAX_VAL]={0}, restricted_hist[MAX_VAL]={0};

//--------------------------------------------------------------------------------
//--------------------------------------------------------------------------------
void record_statistics(int *p, int n) {
    int runs=1, e=1, inverse[MAX_VAL];
    for (int k=0;k<n;++k) {
        inverse[p[k]]=k;
        if(p[k]==e) ++e;
    }
    int first=e-2;
    for(int k=2;k<n;++k)
        if ((inverse[k]>inverse[k-1]) == (inverse[k]>inverse[k+1])) ++runs;
    ++runs_hist[runs]; ++first_hist[first];
    if (p[1]!=2) ++restricted_hist[first];
    ++total;
}
void Print() {
    int a[MAX_VAL], p[MAX_VAL], j=0, i=perm.head;
    while (j<N) { a[j++]=interval[i].right; i=interval[i].next_perm; }
    p[0]=1;
    for (j=0; j<N; ++j) p[j+1]=a[j]+1;
    record_statistics(p,N+1);

}
//--------------------------------------------------------------------------------
void InsertInterval(List *list, int t, int left, int right, int p, int n, int node_ptr) {

    if (p != NULL) interval[p].next = t;
    else list->head = t;

    if (n != NULL) interval[n].prev = t;
    else list->tail = t;

    interval[t].left = left;
    interval[t].right = right;
    interval[t].prev = p;
    interval[t].next = n;
    interval[t].node_ptr = node_ptr;
}

void RemoveInterval(List *list, int t) {

    if (interval[t].next == NULL) list->tail = interval[t].prev;
    else interval[interval[t].next].prev = NULL;

    if (interval[t].prev == NULL) list->head = interval[t].next;
    else interval[interval[t].prev].next = NULL;
}

void MoveInterval(int t, List *list1, List *list2, int p, int n, int node_ptr) {

    RemoveInterval(list1, t);
    InsertInterval(list2, t, interval[t].left, interval[t].right, p, n, node_ptr);
}
//--------------------------------------------------------------------------------
void SetNode(int t, int up_interval, int up_side) {

    node[t].up_interval = up_interval;
    node[t].up_side = up_side;
}

void SetNodeLists(int t, int left_head, int left_tail, int right_head, int right_tail) {

    if (left_head == NULL || left_tail == NULL) left_head = left_tail =  NULL;
    if (right_head == NULL || right_tail == NULL) right_head = right_tail = NULL;

    node[t].L.head = left_head;
    node[t].L.tail = left_tail;
    node[t].R.head = right_head;
    node[t].R.tail=  right_tail;
}
//--------------------------------------------------------------------------------
void UpdatePerm(int t, int i) {
    int p, n;

    p = interval[i].prev_perm;
    n = interval[i].next_perm;

    interval[2*t-1].prev_perm = p;
    interval[2*t-1].next_perm = 2*t;
    interval[2*t].prev_perm = 2*t-1;
    interval[2*t].next_perm = n;

    if (p != NULL) interval[p].next_perm = 2*t-1;
    else perm.head = 2*t-1;
    if (n != NULL) interval[n].prev_perm = 2*t;
}

void RestorePerm(int t, int i) {
    (void)t;
    int p, n;

    p = interval[i].prev_perm;
    n = interval[i].next_perm;

    if (p != NULL) interval[p].next_perm = i;
    else perm.head = i;
    if (n != NULL) interval[n].prev_perm = i;
}
//--------------------------------------------------------------------------------
void Gen(int t, Node* X, int depth) {
    Node *Y;
    int i, up, j, n, p, old_up_side, old_up_interval, side;

    if (t > N) Print();
    else {
        // VISIT LEFT LIST, THEN RIGHT LIST
        for (side=1; side<=2; side++) {

            if (SYM_SEMI && t == 2 && side == 2) return;

            up = 0;
            if (side == 1) i = X->L.head;
            else  i = X->R.head;

            // OPTIMIZATION FOR MEANDERS
            if (MEANDER && N-t <= depth) {
                i = X->up_interval;
                if (X->up_side == 'l')  side = 1;
                else side = 2;
            }

            // VISIT ALL INTERVALS IN LIST
            while (i != NULL) {

                n = interval[i].next;
                p = interval[i].prev;
                j = interval[i].node_ptr;
                Y = &node[j];

                old_up_interval = Y->up_interval;
                old_up_side = Y->up_side;

                SetNode(2*t-1, -1,-1);
                SetNode(2*t, -1,-1);

                // UPDATE NEXT NODE and CREATE 2 NEW NODES
                if (side == 1) {
                    if (STAMP && depth == 0 && i == X->L.head) {
                        up = 1;
                        if (Y->L.head != NULL) SetNodeLists(j, NULL, NULL, Y->L.head, Y->L.tail);
                        MoveInterval(X->R.tail, &X->R, &Y->R, Y->R.tail, NULL, 2*t-1);
                    }
                    else if (X->up_interval == i)  {
                        up = 1;
                        if ((MEANDER || SEMI) && depth == 0) SetNode(j, 2*t-1, 'l');
                    }
                    else if (up == 1 ||  X->up_side == 'r') {
                        SetNode(j, 2*t-1, 'l');
                        SetNode(2*t-1, X->up_interval, X->up_side);
                    }
                    else {
                        SetNode(j, 2*t, 'r');
                        SetNode(2*t, X->up_interval, 'r');
                    }
                    SetNodeLists(2*t-1, X->L.head, p, X->R.head, X->R.tail);
                    SetNodeLists(2*t, NULL, NULL, n, X->L.tail);
                }
                else {
                    if (STAMP && depth == 0 && i == X->R.tail)  {
                        up = 1;
                        if (Y->R.head != NULL) SetNodeLists(j, Y->R.head, Y->R.tail, NULL, NULL);
                        MoveInterval(X->L.head, &X->L, &Y->L, NULL, Y->L.head, 2*t);
                    }
                    else if (X->up_interval == i)  {
                        up = 1;
                        if ((MEANDER || SEMI) && depth == 0) SetNode(j, 2*t, 'r');
                    }
                    else if (up == 1) {
                        SetNode(j, 2*t-1, 'l');
                        SetNode(2*t-1, X->up_interval, 'l');
                    }
                    else {
                        SetNode(j, 2*t, 'r');
                        SetNode(2*t, X->up_interval, X->up_side);
                    }
                    SetNodeLists(2*t-1, X->R.head, p, NULL, NULL);
                    SetNodeLists(2*t, X->L.head, X->L.tail, n, X->R.tail);
                }

                // INSERT NEW INTERVALS
                InsertInterval(&Y->L, 2*t-1, interval[i].left, t, Y->L.tail, NULL, 2*t-1);
                InsertInterval(&Y->R, 2*t, t, interval[i].right, NULL, Y->R.head, 2*t);

                // REMOVE CURRENT INTERVAL
                if (n != NULL) interval[n].prev = NULL;
                if (p != NULL) interval[p].next = NULL;

                // UPDATE THE PERMUTATION
                UpdatePerm(t,i);

                // MAKE RECURSIVE CALLS
                if (STAMP && depth == 0 && (i== X->R.tail || i == X->L.head)) Gen(t+1, Y, 0);
                else if (X->up_interval == i)  Gen(t+1, Y, MAX(0,depth-1));
                else Gen(t+1, Y, depth+1);

                // RESTORE DATA STRUCTURES
                RestorePerm(t,i);

                if (n != NULL) interval[n].prev = i;
                if (p != NULL) interval[p].next = i;

                RemoveInterval(&Y->L, 2*t-1);
                RemoveInterval(&Y->R, 2*t);

                SetNode(j, old_up_interval, old_up_side);

                if (STAMP && depth == 0) {
                    if (i == X->L.head) MoveInterval(Y->R.tail, &Y->R, &X->R, X->R.tail, NULL, j);
                    if (i == X->R.tail) MoveInterval(Y->L.head, &Y->L, &X->L, NULL, X->L.head, j);
                }

                i = n;	// NEXT INTERVAL
                if (MEANDER && N-t <= depth) return;
            }	}	}	}
//--------------------------------------------------------------------------------
int main(int argc, char **argv) {
    int i;

    if (argc!=2) {
        fprintf(stderr, "usage: permutationStatistics n (3<=n<=32)\n");
        return 2;
    }
    N=atoi(argv[1])-1;
    if (N<2 || N>31) {
        fprintf(stderr, "supported indices: 3..32\n");
        return 2;
    }

    // INITIALIZE NODES
    for (i=0; i<=2*N; i++) {
        node[i].L.head = node[i].L.tail = NULL;
        node[i].R.head = node[i].R.tail = NULL;
        node[i].up_interval = NULL;
        node[i].up_side = '-';
    }

    SetNode(0, 2, 'r');
    InsertInterval(&node[0].L, 1, 0, 1, NULL, NULL, 2);
    InsertInterval(&node[0].R, 2, 1, MAX_VAL, NULL, NULL, 1);

    perm.head = 1;
    interval[1].prev_perm = NULL;
    interval[1].next_perm = 2;
    interval[2].prev_perm = 1;
    interval[2].next_perm = NULL;

    Gen(2, &node[0], 0);
    printf("{\"n\":%d,\"total\":%" PRIu64, N+1, total);
    uint64_t *histograms[3]={runs_hist,first_hist,restricted_hist};
    const char *names[3]={"runs","first","restricted"};
    for (int h=0; h<3; ++h) {
        printf(",\"%s\":[",names[h]);
        for (int k=1; k<=N; ++k) printf("%s%" PRIu64,k==1?"":",",histograms[h][k]);
        printf("]");
    }
    printf("}\n");
    return 0;
}

