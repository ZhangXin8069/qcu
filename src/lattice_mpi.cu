#include "../include/qcu.h"
#pragma optimize(5)
namespace qcu
{
    template <>
    int _MPI_Isend<double>(const void *buf, int count, int dest,
                           int tag, MPI_Comm comm, MPI_Request *request)
    {
        return MPI_Isend(buf, count, MPI_DOUBLE, dest, tag, comm, request);
    }
    template <>
    int _MPI_Irecv<double>(void *buf, int count, int source,
                           int tag, MPI_Comm comm, MPI_Request *request)
    {
        return MPI_Irecv(buf, count, MPI_DOUBLE, source, tag, comm, request);
    }
    template <>
    int _MPI_Sendrecv<double>(const void *sendbuf, int sendcount,
                              int dest, int sendtag, void *recvbuf, int recvcount,
                              int source, int recvtag,
                              MPI_Comm comm, MPI_Status *status)
    {
        return MPI_Sendrecv(sendbuf, sendcount, MPI_DOUBLE, dest, sendtag, recvbuf, recvcount, MPI_DOUBLE, source, recvtag, comm, status);
    }
    template <>
    int _MPI_Allreduce<double>(const void *sendbuf, void *recvbuf, int count, MPI_Op op, MPI_Comm comm)
    {
        return MPI_Allreduce(sendbuf, recvbuf, count, MPI_DOUBLE, op, comm);
    }
}