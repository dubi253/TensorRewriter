import { component$, useSignal, useComputed$, useTask$, $, useOn } from '@builder.io/qwik';
import type { DocumentHead } from '@builder.io/qwik-city';
import { Graph, type GraphData } from '~/components/visualizer/graph';

export default component$(() => {
  const query = useSignal('');
  const page = useSignal(0);
  const pageSize = useSignal(6);
  const rules = useSignal<any[]>([]);
  const isLoading = useSignal(true);

  // Fetch rules data on client side to avoid bundling large JSON
  useOn('qvisible', $(async () => {
    if (rules.value.length > 0) return;
    try {
      const res = await fetch(import.meta.env.BASE_URL + 'rules.json');
      rules.value = await res.json();
    } catch (e) {
      console.error('Failed to load rules', e);
    } finally {
      isLoading.value = false;
    }
  }));

  // Scroll to top when page changes
  useTask$(({ track }) => {
    track(() => page.value);
    if (typeof window !== 'undefined') {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  });

  const filteredRules = useComputed$(() => {
    const q = query.value.toLowerCase();
    if (!q) return rules.value;
    return rules.value.filter((r: any) => {
      const sourceOps = r.source.operators.map((o: any) => o.type.toLowerCase()).join(' ');
      const targetOps = r.target.operators.map((o: any) => o.type.toLowerCase()).join(' ');
      return sourceOps.includes(q) || targetOps.includes(q);
    });
  });

  const paginatedRules = useComputed$(() => {
    const start = page.value * pageSize.value;
    return filteredRules.value.slice(start, start + pageSize.value);
  });

  const allOps = useComputed$(() => {
    const ops = new Set<string>();
    rules.value.forEach((r: any) => {
        r.source.operators.forEach((o: any) => ops.add(o.type));
        r.target.operators.forEach((o: any) => ops.add(o.type));
    });
    return Array.from(ops).sort();
  });

  return (
    <div class="min-h-screen bg-base-200 flex flex-col">
      <div class="navbar bg-base-100 shadow-md sticky top-0 z-50 px-4 lg:px-8">
        <div class="flex-1">
          <h1 class="text-2xl font-bold text-primary tracking-tight">Tensor Rewriter</h1>
        </div>
        <div class="flex-none gap-2">
            <div class="join">
                <input
                    type="text"
                    // @ts-expect-error list attribute is valid for input
                    list="ops-list"
                    placeholder="Search rules..."
                    value={query.value}
                    onInput$={(e) => {
                        query.value = (e.target as HTMLInputElement).value;
                        page.value = 0;
                    }}
                    class="input input-bordered join-item w-48 md:w-80 focus:outline-none focus:border-primary"
                />
                <div class="btn no-animation join-item bg-base-200 border-base-300 cursor-default hover:bg-base-200">
                    {filteredRules.value.length}
                </div>
            </div>
            <datalist id="ops-list">
                {allOps.value.map(op => <option value={op} key={op} />)}
            </datalist>
        </div>
      </div>

      <div class="p-4 lg:p-6 w-full max-w-[1920px] mx-auto">
          {isLoading.value ? (
            <div class="flex justify-center items-center h-[60vh]">
                <span class="loading loading-spinner loading-lg text-primary"></span>
            </div>
          ) : (
            <>
              <div class="flex flex-col sm:flex-row justify-end items-center mb-6 gap-4">
                 <div class="flex items-center gap-2">
                    <span class="text-sm text-base-content whitespace-nowrap">Items per page:</span>
                    <select 
                        class="select select-bordered select-sm"
                        value={pageSize.value}
                        onChange$={(e) => {
                            pageSize.value = parseInt((e.target as HTMLSelectElement).value);
                            page.value = 0;
                        }}
                    >
                        <option value="6">6</option>
                        <option value="10">10</option>
                        <option value="20">20</option>
                        <option value="50">50</option>
                    </select>
                 </div>

                 <div class="join shadow-sm">
                    <button 
                        class="join-item btn btn-sm" 
                        onClick$={() => page.value = Math.max(0, page.value - 1)} 
                        disabled={page.value === 0}
                    >
                        «
                    </button>
                    <button class="join-item btn btn-sm no-animation bg-base-100">
                        Page {page.value + 1} of {Math.max(1, Math.ceil(filteredRules.value.length / pageSize.value))}
                    </button>
                    <button 
                        class="join-item btn btn-sm" 
                        onClick$={() => page.value = Math.min(Math.ceil(filteredRules.value.length / pageSize.value) - 1, page.value + 1)} 
                        disabled={page.value >= Math.ceil(filteredRules.value.length / pageSize.value) - 1}
                    >
                        »
                    </button>
                </div>
              </div>

              <div class="grid grid-cols-1 xl:grid-cols-2 gap-6">
                {paginatedRules.value.map((rule: any) => (
                  <div key={rule.id} class="card bg-base-100 shadow-md hover:shadow-xl transition-all duration-300 border border-base-200">
                    <div class="card-body p-6">
                        <div class="flex justify-between items-center">
                            <h3 class="card-title text-lg font-bold text-base-content">Rule {rule.id}</h3>
                        </div>
                        
                        <div class="divider my-0"></div>

                        <div class="flex flex-col lg:flex-row items-center gap-0 w-full">
                            <div class="flex-1 w-full flex flex-col items-center p-2">
                                <span class="text-xs font-bold uppercase tracking-wider text-base-content/50 mb-2">Source</span>
                                <div class="w-full bg-base-50/50 rounded-xl border border-base-200/50 overflow-hidden">
                                     <Graph data={rule.source as GraphData} />
                                </div>
                            </div>
                            
                            <div class="text-2xl text-base-content/20 rotate-90 lg:rotate-0 p-2">
                                ➜
                            </div>

                            <div class="flex-1 w-full flex flex-col items-center p-2">
                                <span class="text-xs font-bold uppercase tracking-wider text-base-content/50 mb-2">Target</span>
                                <div class="w-full bg-base-50/50 rounded-xl border border-base-200/50 overflow-hidden">
                                    <Graph data={rule.target as GraphData} />
                                </div>
                            </div>
                        </div>
                    </div>
                  </div>
                ))}
              </div>
              
              <div class="flex justify-center mt-10 mb-4">
                 <div class="join shadow-md">
                    <button 
                        class="join-item btn" 
                        onClick$={() => page.value = Math.max(0, page.value - 1)} 
                        disabled={page.value === 0}
                    >
                        « Previous
                    </button>
                    <button class="join-item btn no-animation bg-base-100 w-32">
                        Page {page.value + 1}
                    </button>
                    <button 
                        class="join-item btn" 
                        onClick$={() => page.value = Math.min(Math.ceil(filteredRules.value.length / pageSize.value) - 1, page.value + 1)} 
                        disabled={page.value >= Math.ceil(filteredRules.value.length / pageSize.value) - 1}
                    >
                        Next »
                    </button>
                </div>
              </div>
            </>
          )}
      </div>
    </div>
  );
});

export const head: DocumentHead = {
  title: "Tensor Rewriter Rules",
  meta: [
    {
      name: "description",
      content: "Visualization of tensor rewriting rules",
    },
  ],
};
