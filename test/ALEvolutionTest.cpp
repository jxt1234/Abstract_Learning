//#include "test/GPTest.h"
//#include "core/GPProducer.h"
//#include "core/GPFactory.h"
//#include "system/system_lib.h"
//#include "evolution/GPEvolutionGroup.h"
//#include <iostream>
//#include <sstream>
//#include <fstream>
//#include <assert.h>
//#include <string>
//using namespace std;
//class ALEvolutionTest:public GPTest
//{
//    public:
//        virtual void run()
//        {
//            srand((unsigned) time(NULL));
//            system_lib table(string("libAbstract_learning"));
//            GPFunctionDataBase* base = GPFactory::createDataBase("Learn.xml", &table);
//            AUTOCLEAN(base);
//            /*FitComputer*/
//            const IStatusType* learner = base->vQueryType("ALISuperviseLearner");
//            const IStatusType* dcId = base->vQueryType("ALLabeldData");
//            const IStatusType* output_double = base->vQueryType("double");
//            const IStatusType* charType = base->vQueryType("char");
//            GP_Output dataChain;
//            GPProducer* sys = GPFactory::createProducer(base);
//            AUTOCLEAN(sys);
//            void* chain;
//            {
//                vector<const IStatusType*> input(1, charType);
//                vector<const IStatusType*> output(1, dcId);
//                IGPAutoDefFunction* loader = sys->vCreateFunction(output, input);
//                istringstream is("bao.txt");
//                void* fileName = charType->vLoad(is);
//                vector<void*> inp;
//                inp.push_back(fileName);
//                dataChain = loader->run(inp);
//                chain = dataChain[0];
//                charType->vFree(fileName);
//            }
//            IGPAutoDefFunction* fit = NULL;
//            {
//                vector<const IStatusType*> input;
//                input.push_back(learner);
//                input.push_back(dcId);
//                vector<const IStatusType*> output;
//                output.push_back(output_double);
//                IGPAutoDefFunction* fit_inner = sys->vCreateFunction(output, input);
//                class fitC:public IGPAutoDefFunction
//                {
//                    public:
//                        fitC(IGPAutoDefFunction* f, void* c):mI(c), mF(f){}
//                        ~fitC(){}
//                        virtual GP_Output run(const GP_Input& input)
//                        {
//                            GP_Input input2 = input;
//                            input2.push_back(mI);
//                            return mF->run(input2);
//                        }
//                        virtual IGPAutoDefFunction* copy() const
//                        {
//                            return NULL;
//                        }
//                        virtual int vMap(GPPtr<GPParameter> para)
//                        {
//                            return 0;
//                        }
//                        virtual std::vector<const IStatusType*> vGetInputs() const
//                        {
//                        }
//                        /*Return all outputTypes in order*/
//                        virtual std::vector<const IStatusType*> vGetOutputs() const
//                        {
//                        }
//                    private:
//                        void* mI;
//                        IGPAutoDefFunction* mF;
//                };
//                fit = new fitC(fit_inner, chain);
//            }
//            GPEvolutionGroup* group = new GPEvolutionGroup(sys, 10, 10);
//            {
//                GP_Input nullInput;
//                vector<const IStatusType*> input;
//                vector<const IStatusType*> output(1, learner);
//                group->vSetInput(input);
//                group->vSetOutput(output);
//                group->vSetFixInput(nullInput);
//            }
//            group->vEvolution(fit);
//            fit->decRef();
//            IGPAutoDefFunction* res = group->getBest();
//            res->addRef();
//            cout << "Best Fit is "<<group->getBestFit()<<endl;
//            delete group;
//
//            ofstream outputF("output/ALEvolutionTest.xml");
//            res->save(outputF);
//            outputF.close();
//            res->decRef();
//            GP_Output_clear(dataChain);
//        }
//        ALEvolutionTest(){}
//        virtual ~ALEvolutionTest(){}
//};
//
//static GPTestRegister<ALEvolutionTest> a("ALEvolutionTest");
