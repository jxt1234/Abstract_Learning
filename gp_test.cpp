#include "user/GPAPI.h"
#include <stdlib.h>
using namespace std;


static void __run()
{
    GPStream* metaStream = GP_Stream_Create("./libAbstract_learning.xml");
    auto producer = GP_Producer_Create(&metaStream, NULL, 1, GP_PRODUCER_STREAM);
    GP_Stream_Destroy(metaStream);

    metaStream = GP_Stream_Create("./mgpfunc.xml");
    auto map_reduce = GP_Stream_Create("Map-Reduce.xml");
    auto pieceProducer = GP_PiecesProducer_Create(producer, &metaStream, NULL, 1, &map_reduce, 1);
    GP_Stream_Destroy(metaStream);
    GP_Stream_Destroy(map_reduce);

    GP_PiecesProducer_Destroy(pieceProducer);
    GP_Producer_Destroy(producer);
}

int main()
{
    __run();
    return 1;
}
