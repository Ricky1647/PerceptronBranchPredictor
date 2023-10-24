/*
 * Copyright (c) 2004-2006 The Regents of The University of Michigan
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "cpu/pred/perceptron.hh"

#include "base/intmath.hh"
#include "base/logging.hh"
#include "base/trace.hh"
#include "debug/Fetch.hh"
#include "cmath"

namespace gem5
{

namespace branch_prediction
{

PerceptronBP::PerceptronBP(const PerceptronBPParams &params)
    : BPredUnit(params),
      PredictorEntry(params.PredictorEntry),
      HistoryLen(params.HistoryLen),
      Threshold(int(1.93 * HistoryLen + 14)), 
      WeightBits(int(1 + log2(Threshold))),
      PredictorSize(PredictorEntry * WeightBits * (HistoryLen + 1) + HistoryLen * params.numThreads),
      GlobalTable(PredictorEntry * (HistoryLen + 1)), //[(HistoryLen+1)*i]: Newest, [(HistoryLen+1)*i+HistoryLen-1]: Oldest, [(HistoryLen+1)*i+HistoryLen]: Bias
      GlobalHistory(params.numThreads * HistoryLen)  // [HistoryLen*i]: Newest, [HistoryLen*i+HistoryLen-1]: Oldest
{

    DPRINTF(Fetch, "number of thread: %i\n",
            params.numThreads);
    DPRINTF(Fetch, "number of entry: %i\n",
            PredictorEntry);
    DPRINTF(Fetch, "history length: %i\n",
            HistoryLen);
    DPRINTF(Fetch, "threshold: %i\n",
            Threshold);
    DPRINTF(Fetch, "weight bits: %i\n", WeightBits);
    DPRINTF(Fetch, "predictor size: %i\n",
            PredictorSize);

    DPRINTF(Fetch, "instruction shift amount: %i\n",
            instShiftAmt);

    for(int i = 0; i < GlobalHistory.size(); i++)
        GlobalHistory[i] = -1;
}

void
PerceptronBP::btbUpdate(ThreadID tid, Addr branch_addr, void * &bp_history)
{
    //Update Global History to Not Taken (clear LSB)
    GlobalHistory[tid * HistoryLen] = -1;
}


bool
PerceptronBP::lookup(ThreadID tid, Addr branch_addr, void * &bp_history)
{
    bool taken;
    unsigned predictor_idx = getTableIndex(branch_addr);

    DPRINTF(Fetch, "Looking up index %#x\n",
            predictor_idx);

    std::vector<int> WeightArr(GlobalTable.begin() + (HistoryLen + 1) * predictor_idx, GlobalTable.begin() + (HistoryLen + 1) * (predictor_idx + 1));

    int y_out = WeightArr[WeightArr.size() - 1];
    for(int i = 0; i < WeightArr.size() - 1; i++) {
        y_out += WeightArr[i] * GlobalHistory[tid * HistoryLen + i];
    }

    DPRINTF(Fetch, "prediction is %i.\n",
            y_out);

    taken = (y_out > 0);

    BPHistory *history = new BPHistory{
        std::vector<int>(HistoryLen, -1),
        true,
        0,
        0
    };

    for(int i = 0; i < HistoryLen; i++) {
        history->GlobalHistory[i] = GlobalHistory[tid * HistoryLen + i];
    }
    history->predictor_idx = predictor_idx;
    history->isConditional = true;
    history->y_out_cache = y_out;
    bp_history = (void *)history;

    // Speculative update of the global history
    for(int i = HistoryLen - 1; i >= 0; i--) {
        if(taken) {
            if(i == 0) GlobalHistory[tid * HistoryLen + i] = 1;
            else GlobalHistory[tid * HistoryLen + i] = GlobalHistory[tid * HistoryLen + i - 1];
        }
        else {
            if(i == 0) GlobalHistory[tid * HistoryLen + i] = -1;
            else GlobalHistory[tid * HistoryLen + i] = GlobalHistory[tid * HistoryLen + i - 1];
        }
    }    

    return taken;
}

void
PerceptronBP::update(ThreadID tid, Addr branch_addr, bool taken, void *bp_history,
                bool squashed, const StaticInstPtr & inst, Addr corrTarget)
{
    assert(bp_history);

    BPHistory *history = static_cast<BPHistory *>(bp_history);

    // No state to restore, and we do not update on the wrong
    // path.
    if (squashed) {
        for(int i = 0; i < HistoryLen; i++) {
            GlobalHistory[tid * HistoryLen + i] = history->GlobalHistory[i];
        }
        return;
    }

    // Update the predictor.
    unsigned old_predictor_idx = history->predictor_idx;

    DPRINTF(Fetch, "Looking up index %#x\n", old_predictor_idx);

    bool sign = (history->y_out_cache > 0); 
    if (taken) {
        DPRINTF(Fetch, "Branch updated as taken.\n");
        if(sign != taken || abs(history->y_out_cache) <= Threshold) {
            for(int i = 0; i < HistoryLen; i++)
                GlobalTable[old_predictor_idx * (HistoryLen + 1) + i] += GlobalHistory[tid * HistoryLen + i];
            GlobalTable[old_predictor_idx * (HistoryLen + 1) + HistoryLen] += 1;
        }
    } else {
        DPRINTF(Fetch, "Branch updated as not taken.\n");
        if(sign != taken || abs(history->y_out_cache) <= Threshold) {
            for(int i = 0; i < HistoryLen; i++)
                GlobalTable[old_predictor_idx * (HistoryLen + 1) + i] -= GlobalHistory[tid * HistoryLen + i];
            GlobalTable[old_predictor_idx * (HistoryLen + 1) + HistoryLen] -= 1;            
        }        
    }
    delete history;
}

inline
unsigned
PerceptronBP::getTableIndex(Addr &branch_addr)
{
    return (branch_addr >> instShiftAmt) % PredictorEntry;
}

void
PerceptronBP::uncondBranch(ThreadID tid, Addr pc, void *&bp_history)
{
    // Create BPHistory and pass it back to be recorded.
    BPHistory *history = new BPHistory{
        std::vector<int>(HistoryLen, -1),
        false,
        0,
        0
    };
    for(int i = 0; i < HistoryLen; i++) {
        history->GlobalHistory[i] = GlobalHistory[tid * HistoryLen + i];
    }
//////////////////////////////////////////////////
    unsigned predictor_idx = getTableIndex(pc);
    std::vector<int> WeightArr(GlobalTable.begin() + (HistoryLen + 1) * predictor_idx, GlobalTable.begin() + (HistoryLen + 1) * (predictor_idx + 1));

    int y_out = WeightArr[WeightArr.size() - 1];
    for(int i = 0; i < WeightArr.size() - 1; i++) {
        y_out += WeightArr[i] * GlobalHistory[tid * HistoryLen + i];
    }
//////////////////////////////////////////////////////////
    history->predictor_idx = getTableIndex(pc);
    history->isConditional = false;
    history->y_out_cache = y_out;
    bp_history = static_cast<void *>(history);
    for(int i = HistoryLen - 1; i >= 0; i--) {
        if(i == 0) GlobalHistory[tid * HistoryLen + i] = 1;
        else GlobalHistory[tid * HistoryLen + i] = GlobalHistory[tid * HistoryLen + i - 1];
    }   
}

void
PerceptronBP::squash(ThreadID tid, void *bp_history)
{
    BPHistory *history = static_cast<BPHistory *>(bp_history);
    // Restore global history to state prior to this branch.
    for(int i = 0; i < HistoryLen; i++) {
        GlobalHistory[tid * HistoryLen + i] = history->GlobalHistory[i];
    }
    // Delete this BPHistory now that we're done with it.
    delete history;
}

} // namespace branch_prediction
} // namespace gem5