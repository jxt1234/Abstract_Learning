#include "learn/ALStepRegressor.h"
#include "learn/ALRegressor.h"
#include "learn/ALLearnFactory.h"
#include <algorithm>
class ALStepExpander:public ALIExpander
{
    public:
        ALStepExpander(std::vector<int> ar, std::vector<int> wd, bool setConst)
        {
            mLength = (int)ar.size() + (int)wd.size();
            ALASSERT(mLength>0);
            mAR = ar;
            mWd = wd;
            std::sort(mAR.begin(), mAR.end());
            std::sort(mWd.begin(), mWd.end());
            ALASSERT(mAR.empty() || mAR[0]!=0);
            mC = setConst;
            if (mC)
            {
                mLength+=1;
            }
        }
        virtual ~ALStepExpander()
        {
        }
        virtual int vLength() const override
        {
            return mLength;
        }
        virtual bool vExpand(const ALFloatData* d, ALFLOAT* dst) const override
        {
            int dstCur = 0;
            /*Check For AR*/
            if (!mAR.empty())
            {
                int maxAR = mAR[mAR.size()-1];
                if (!d->canBack(maxAR))
                {
                    return false;
                }
            }
            /*Expand WD Firstly*/
            for (int i=0; i<mWd.size(); ++i)
            {
                dst[dstCur++] = d->value(mWd[i]);
            }
            /*Expand AR Secondly*/
            int curBack = 0;
            for (int i=0; i<mAR.size(); ++i)
            {
                int b = mAR[i];
                while (curBack < b)
                {
                    d = d->front();
                    curBack++;
                }
                dst[dstCur++] = d->value(0);
            }

            if (mC)
            {
                dst[dstCur++] = 1.0f;
            }
            return true;
        }
        virtual void vPrint(std::ostream& os) const override
        {
            os << "<Step>\n";
            os << "<AR>";
            for (int i=0; i<mAR.size(); ++i)
            {
                os << mAR[i]<<" ";
            }
            os << "</AR>\n";
            os << "<WD>";
            for (int i=0; i<mWd.size(); ++i)
            {
                os << mWd[i]<<" ";
            }
            os << "</WD>\n";
            os << "<Const>"<<mC << "</Const>\n";
            os << "</Step>\n";
        }
    private:
        std::vector<int> mAR;
        std::vector<int> mWd;
        int mLength;
        bool mC;
};

ALIExpander* ALStepRegressor::train(const ALLabeldData* data) const
{
    ALASSERT(NULL!=data);
    ALASSERT(data->get().size()>10);
    std::vector<int> ar;
    std::vector<int> wd(1, 0);
    bool setConst = false;
    size_t maxW = data->get()[0].second->num();

    ALSp<ALIExpander> xe = new ALStepExpander(ar, wd, setConst);
    ALSp<ALISuperviseLearner> basic = new ALRegressor;
    ALSp<ALIChainLearner> learner = ALIChainLearner::createFromBasic(basic, xe);
    double maxV = ALLearnFactory::crossValidate(learner.get(), data);

    {
        learner = ALIChainLearner::createFromBasic(basic, new ALStepExpander(ar, wd, true));
        double v = ALLearnFactory::crossValidate(learner.get(), data);
        if (maxV < v)
        {
            setConst = true;
            maxV = v;
        }
    }

    for (int i=1; i<maxW; ++i)
    {
        wd.push_back(i);
        learner = ALIChainLearner::createFromBasic(basic, new ALStepExpander(ar, wd, setConst));
        double v = ALLearnFactory::crossValidate(learner.get(), data);
        if (maxV < v)
        {
            maxV = v;
        }
        else
        {
            wd.pop_back();
        }
    }
    for (int i=1; i<mMaxBack; ++i)
    {
        ar.push_back(i);
        learner = ALIChainLearner::createFromBasic(basic, new ALStepExpander(ar, wd, setConst));
        double v = ALLearnFactory::crossValidate(learner.get(), data);
        if (maxV < v)
        {
            maxV = v;
        }
        else
        {
            ar.pop_back();
        }
    }
    return new ALStepExpander(ar, wd, setConst);
}


ALFloatPredictor* ALStepRegressor::vLearn(const ALLabeldData* data) const
{
    ALASSERT(NULL!=data);
    if (data->get().size()<=10)
    {
        return new ALDummyFloatPredictor;
    }
    ALSp<ALIExpander> xe = train(data);
    ALSp<ALIChainLearner> l = ALIChainLearner::createFromBasic(new ALRegressor, xe);
    return l->vLearn(data);
}
