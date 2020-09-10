#include <math.h>
#include <vector>
#include <fstream>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>

#include <xtensor/xio.hpp>

#include <mure/mure.hpp>
#include "coarsening.hpp"
#include "refinement.hpp"
#include "criteria.hpp"

#include <chrono>


/// Timer used in tic & toc
auto tic_timer = std::chrono::high_resolution_clock::now();

/// Launching the timer
void tic()
{
    tic_timer = std::chrono::high_resolution_clock::now();
}


/// Stopping the timer and returning the duration in seconds
double toc()
{
    const auto toc_timer = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> time_span = toc_timer - tic_timer;
    return time_span.count();
}

std::array<double, 3> exact_solution(double x, double t)   {

    double density  = 0.0;
    double velocity = 0.0;
    double pressure = 0.0;

    if (t <= 0.0){
        density  = (x <= 0.0) ? 1.0 : 0.125;
        velocity =              0.0;
        pressure = (x <= 0.0) ? 1.0 : 0.1;
    }
    else
    {
        double gm = 1.4;

        double rhoL = 1.0;
        double rhoR = 0.125;
        double uL = 0.0;
        double uR = 0.0;
        double pL = 1.0;
        double pR = 0.1;

        double cL = sqrt(gm * rhoL / pL);
        double cR = sqrt(gm * rhoR / pR);

        double pStar = 0.30313;
        double uStar = 0.92745;

        double cLStar = cL * pow(pStar / pL, (gm-1)/(2*gm));


        double rhoLStar = rhoL * pow(pStar/pL, 1./gm);
        double rhoRStar = rhoR * ((pStar/pR + (gm-1)/(gm+1)) / ((gm-1)/(gm+1) * pStar/pR + 1));

        double xFL = (uL - cL) * t;
        double xFR = (uStar - cLStar) * t;
        double xContact = (uL + 2*cL/(gm-1)*(1 - pow(pStar/pL, (gm-1)/(2*gm)))) * t;
        double xShock = (uR + cR * sqrt((gm+1)/(2*gm)*pStar/pR + (gm-1)/(2*gm))) * t;

        if (x <= xFL)   {
            density = rhoL;
            velocity = uL;
            pressure = pL;
        }
        else
        {
            if (x <= xFR)   {
                density = rhoL * pow(2/(gm+1) + (gm-1)/(cL * (gm+1)) * (uL - x/t), 2/(gm-1));
                velocity = 2./(gm+1)*(cL + (gm-1)/2*uL + x/t);
                pressure = pL*pow(2/(gm+1)+(gm-1)/((gm+1)*cL)*(uL - x/t), 2*gm/(gm-1));
            }
            else
            {
                if (x <= xContact){
                    density = rhoLStar;
                    velocity = uStar;
                    pressure = pStar;
                }
                else
                {
                    if (x <= xShock){
                        density = rhoRStar;
                        velocity = uStar;
                        pressure = pStar;
                    }
                    else
                    {
                        density = rhoR;
                        velocity = uR;
                        pressure = pR;
                    }   
                }   
            }   
        }
    }
    
    return {density, velocity, pressure};
}


template<class Config>
auto init_f(mure::Mesh<Config> &mesh, double t)
{
    constexpr std::size_t nvel = 6;
    mure::BC<1> bc{ {{ {mure::BCType::neumann, 0.0},
                       {mure::BCType::neumann, 0.0},
                       {mure::BCType::neumann, 0.0},
                       {mure::BCType::neumann, 0.0},
                       {mure::BCType::neumann, 0.0},
                       {mure::BCType::neumann, 0.0},
                    }} };

    mure::Field<Config, double, nvel> f("f", mesh, bc);
    f.array().fill(0);

    double gamma = 1.4;

    mesh.for_each_cell([&](auto &cell) {
        auto center = cell.center();
        auto x = center[0];
        
        
        auto initial_data = exact_solution(x, 0.0);


        double density  = initial_data[0];
        double velocity = initial_data[1];
        double pressure = initial_data[2];

        double u10 = density;
        double u20 = density * velocity;
        double u30 = 0.5 * density * velocity * velocity + pressure / (gamma - 1.0);

        double u11 = u20;
        double u21 = (gamma - 1.0) * u30 + (3.0 - gamma)/(2.0) * (u20*u20)/u10; 
        double u31 = gamma * (u20*u30)/(u10) + (1.0 - gamma)/2.0 * (u20*u20*u20)/(u10*u10);

        double lambda = 3.0;

        f[cell][0] = .5 * (u10 + u11/lambda);
        f[cell][1] = .5 * (u10 - u11/lambda);

        f[cell][2] = .5 * (u20 + u21/lambda);
        f[cell][3] = .5 * (u20 - u21/lambda);

        f[cell][4] = .5 * (u30 + u31/lambda);
        f[cell][5] = .5 * (u30 - u31/lambda);
    });

    return f;
}

template<class Field, class interval_t, class FieldTag>
xt::xtensor<double, 1> prediction(const Field& f, std::size_t level_g, std::size_t level, const interval_t &i, const std::size_t item, 
                                  const FieldTag & tag, std::map<std::tuple<std::size_t, std::size_t, std::size_t, interval_t>, 
                                  xt::xtensor<double, 1>> & mem_map)
{

    // We check if the element is already in the map
    auto it = mem_map.find({item, level_g, level, i});
    if (it != mem_map.end())   {
        //std::cout<<std::endl<<"Found by memoization";
        return it->second;
    }
    else {

        auto mesh = f.mesh();
        xt::xtensor<double, 1> out = xt::empty<double>({i.size()/i.step});//xt::eval(f(item, level_g, i));
        auto mask = mesh.exists(level_g + level, i);

        // std::cout << level_g + level << " " << i << " " << mask << "\n"; 
        if (xt::all(mask))
        {         
            return xt::eval(f(item, level_g + level, i));
        }

        auto step = i.step;
        auto ig = i / 2;
        ig.step = step >> 1;
        xt::xtensor<double, 1> d = xt::empty<double>({i.size()/i.step});

        for (int ii=i.start, iii=0; ii<i.end; ii+=i.step, ++iii)
        {
            d[iii] = (ii & 1)? -1.: 1.;
        }

    
        auto val = xt::eval(prediction(f, level_g, level-1, ig, item, tag, mem_map) - 1./8 * d * (prediction(f, level_g, level-1, ig+1, item, tag, mem_map) 
                                                                                       - prediction(f, level_g, level-1, ig-1, item, tag, mem_map)));
        

        xt::masked_view(out, !mask) = xt::masked_view(val, !mask);
        for(int i_mask=0, i_int=i.start; i_int<i.end; ++i_mask, i_int+=i.step)
        {
            if (mask[i_mask])
            {
                out[i_mask] = f(item, level_g + level, {i_int, i_int + 1})[0];
            }
        }

        // The value should be added to the memoization map before returning
        return mem_map[{item, level_g, level, i}] = out;

        //return out;
    }

}


template<class Field, class interval_t>
xt::xtensor<double, 2> prediction_all(const Field& f, std::size_t level_g, std::size_t level, const interval_t &i, 
                                  std::map<std::tuple<std::size_t, std::size_t, interval_t>, 
                                  xt::xtensor<double, 2>> & mem_map)
{

    using namespace xt::placeholders;
    // We check if the element is already in the map
    auto it = mem_map.find({level_g, level, i});
    if (it != mem_map.end())
    {
        return it->second;
    }
    else
    {
        auto mesh = f.mesh();
        std::vector<std::size_t> shape = {i.size(), 6};
        xt::xtensor<double, 2> out = xt::empty<double>(shape);
        auto mask = mesh.exists(level_g + level, i);

        xt::xtensor<double, 2> mask_all = xt::empty<double>(shape);
        xt::view(mask_all, xt::all(), 0) = mask;
        xt::view(mask_all, xt::all(), 1) = mask;
        xt::view(mask_all, xt::all(), 2) = mask;
        xt::view(mask_all, xt::all(), 3) = mask;
        xt::view(mask_all, xt::all(), 4) = mask;
        xt::view(mask_all, xt::all(), 5) = mask;

        if (xt::all(mask))
        {         
            return xt::eval(f(level_g + level, i));
        }

        auto ig = i >> 1;
        ig.step = 1;

        xt::xtensor<double, 2> val = xt::empty<double>(shape);
        auto current = xt::eval(prediction_all(f, level_g, level-1, ig, mem_map));
        auto left = xt::eval(prediction_all(f, level_g, level-1, ig-1, mem_map));
        auto right = xt::eval(prediction_all(f, level_g, level-1, ig+1, mem_map));

        std::size_t start_even = (i.start&1)? 1: 0;
        std::size_t start_odd = (i.start&1)? 0: 1;
        std::size_t end_even = (i.end&1)? ig.size(): ig.size()-1;
        std::size_t end_odd = (i.end&1)? ig.size()-1: ig.size();
        xt::view(val, xt::range(start_even, _, 2)) = xt::view(current - 1./8 * (right - left), xt::range(start_even, _));
        xt::view(val, xt::range(start_odd, _, 2)) = xt::view(current + 1./8 * (right - left), xt::range(_, end_odd));

        xt::masked_view(out, !mask_all) = xt::masked_view(val, !mask_all);
        for(int i_mask=0, i_int=i.start; i_int<i.end; ++i_mask, ++i_int)
        {
            if (mask[i_mask])
            {
                xt::view(out, i_mask) = xt::view(f(level_g + level, {i_int, i_int + 1}), 0);
            }
        }

        // The value should be added to the memoization map before returning
        return out;// mem_map[{level_g, level, i, ig}] = out;
    }
}

template<class Field, class FieldTag>
void one_time_step(Field &f, const FieldTag & tag, double s)
{
    constexpr std::size_t nvel = Field::size;
    double lambda = 3.;//, s = 1.0;
    auto mesh = f.mesh();
    auto max_level = mesh.max_level();
    
    mure::mr_projection(f);
    f.update_bc();
    mure::mr_prediction(f);



    // MEMOIZATION
    // All is ready to do a little bit  of mem...
    using interval_t = typename Field::Config::interval_t;
    std::map<std::tuple<std::size_t, std::size_t, std::size_t, interval_t>, xt::xtensor<double, 1>> memoization_map;
    memoization_map.clear(); // Just to be sure...

    Field new_f{"new_f", mesh};
    new_f.array().fill(0.);

    for (std::size_t level = 0; level <= max_level; ++level)
    {
        auto exp = mure::intersection(mesh[mure::MeshType::cells][level],
                                      mesh[mure::MeshType::cells][level]);
        exp([&](auto, auto &interval, auto) {
            auto i = interval[0];


            // STREAM

            std::size_t j = max_level - level;

            double coeff = 1. / (1 << j);

            // This is the STANDARD FLUX EVALUATION


            auto fp1 = f(0, level, i) + coeff * (prediction(f, level, j, i*(1<<j)-1, 0, tag, memoization_map)
                                             -  prediction(f, level, j, (i+1)*(1<<j)-1, 0, tag, memoization_map));
            //std::cout<<"Plus"<<std::endl;
            auto fm1 = f(1, level, i) - coeff * (prediction(f, level, j, i*(1<<j), 1, tag, memoization_map)
                                             -  prediction(f, level, j, (i+1)*(1<<j), 1, tag, memoization_map));
            //std::cout<<"Minus"<<std::endl;

            auto fp2 = f(2, level, i) + coeff * (prediction(f, level, j, i*(1<<j)-1, 2, tag, memoization_map)
                                             -  prediction(f, level, j, (i+1)*(1<<j)-1, 2, tag, memoization_map));
            auto fm2 = f(3, level, i) - coeff * (prediction(f, level, j, i*(1<<j), 3, tag, memoization_map)
                                             -  prediction(f, level, j, (i+1)*(1<<j), 3, tag, memoization_map));

            auto fp3 = f(4, level, i) + coeff * (prediction(f, level, j, i*(1<<j)-1, 4, tag, memoization_map)
                                             -  prediction(f, level, j, (i+1)*(1<<j)-1, 4, tag, memoization_map));
            auto fm3 = f(5, level, i) - coeff * (prediction(f, level, j, i*(1<<j), 5, tag, memoization_map)
                                             -  prediction(f, level, j, (i+1)*(1<<j), 5, tag, memoization_map));

            //std::cout<<std::endl<<"Done";

           
            auto u10 = xt::eval(          fp1 + fm1);
            auto u11 = xt::eval(lambda * (fp1 - fm1));
           
            auto u20 = xt::eval(          fp2 + fm2);
            auto u21 = xt::eval(lambda * (fp2 - fm2));
           
            auto u30 = xt::eval(          fp3 + fm3);
            auto u31 = xt::eval(lambda * (fp3 - fm3));


            double gamma = 1.4;
            auto u11_coll = (1 - s) * u11 + s * (u20);

            auto u21_coll = (1 - s) * u21 + s * ((gamma - 1.0) * u30 + (3.0 - gamma)/(2.0) * (u20*u20)/u10);

            auto u31_coll = (1 - s) * u31 + s * (gamma * (u20*u30)/(u10) + (1.0 - gamma)/2.0 * (u20*u20*u20)/(u10*u10));


            new_f(0, level, i) = .5 * (u10 + 1. / lambda * u11_coll);
            new_f(1, level, i) = .5 * (u10 - 1. / lambda * u11_coll);

            new_f(2, level, i) = .5 * (u20 + 1. / lambda * u21_coll);
            new_f(3, level, i) = .5 * (u20 - 1. / lambda * u21_coll);

            new_f(4, level, i) = .5 * (u30 + 1. / lambda * u31_coll);
            new_f(5, level, i) = .5 * (u30 - 1. / lambda * u31_coll);

        });
    }

    std::swap(f.array(), new_f.array());
}

template<class Config, class FieldR>
std::array<double, 6> compute_error(mure::Field<Config, double, 6> &f, FieldR & fR, double t)
{

    auto mesh = f.mesh();

    auto meshR = fR.mesh();
    auto max_level = meshR.max_level();

    // mure::mr_projection(f);
    // mure::mr_prediction(f);  // C'est supercrucial de le faire.
    

    fR.update_bc();    

    mure::mr_projection(f);
    
    f.update_bc(); // Important especially when we enforce Neumann...for the Riemann problem

    mure::mr_prediction(f);  // C'est supercrucial de le faire.

    // f.update_bc(); // Important especially when we enforce Neumann...for the Riemann problem
    // fR.update_bc();  

    // Getting ready for memoization
    // using interval_t = typename Field::Config::interval_t;
    using interval_t = typename Config::interval_t;
    std::map<std::tuple<std::size_t, std::size_t, interval_t>, xt::xtensor<double, 2>> error_memoization_map;
    error_memoization_map.clear();

    double error_rho = 0.0; // First momentum 
    double error_q = 0.0; // Second momentum
    double error_E = 0.0; // Third momentum

    double diff_rho = 0.0;
    double diff_q = 0.0;
    double diff_E = 0.0;


    double dx = 1.0 / (1 << max_level);

    for (std::size_t level = 0; level <= max_level; ++level)
    {
        auto exp = mure::intersection(meshR[mure::MeshType::cells][max_level],
                                      mesh[mure::MeshType::cells][level])
                  .on(max_level);

        exp([&](auto, auto &interval, auto) {
            auto i = interval[0];
            auto j = max_level - level;

            auto sol  = prediction_all(f, level, j, i, error_memoization_map);
            auto solR = xt::view(fR(max_level, i), xt::all(), xt::range(0, 3));


            xt::xtensor<double, 1> x = dx*xt::linspace<int>(i.start, i.end - 1, i.size()) + 0.5*dx;


            xt::xtensor<double, 1> rhoexact = xt::zeros<double>(x.shape());
            xt::xtensor<double, 1> qexact = xt::zeros<double>(x.shape());
            xt::xtensor<double, 1> Eexact = xt::zeros<double>(x.shape());

            double gm = 1.4;
            double lambda = 3.0;


            for (std::size_t idx = 0; idx < x.shape()[0]; ++idx)    {
                auto ex_sol = exact_solution(x[idx], t);

                rhoexact[idx] = ex_sol[0];
                qexact[idx] = ex_sol[0]*ex_sol[1];
                Eexact[idx] = 0.5 * ex_sol[0]*pow(ex_sol[1], 2.0) + ex_sol[2] / (gm - 1.);

            }

            error_rho += xt::sum(xt::abs(xt::flatten(xt::view(fR(max_level, i), xt::all(), 0) 
                                                 + xt::view(fR(max_level, i), xt::all(), 1)) 
                                     - rhoexact))[0];

            error_q += xt::sum(xt::abs(xt::flatten(xt::view(fR(max_level, i), xt::all(), 2)
                                                 + xt::view(fR(max_level, i), xt::all(), 3)) 
                                     - qexact))[0];

            error_E += xt::sum(xt::abs(xt::flatten(xt::view(fR(max_level, i), xt::all(), 4)
                                                 + xt::view(fR(max_level, i), xt::all(), 5)) 
                                     - Eexact))[0];


            diff_rho += xt::sum(xt::abs(xt::flatten(xt::view(sol, xt::all(), 0) 
                                                  + xt::view(sol, xt::all(), 1))
                                                - xt::flatten(xt::view(fR(max_level, i), xt::all(), 0) 
                                                            + xt::view(fR(max_level, i), xt::all(), 1)))) [0];
            
            diff_q += xt::sum(xt::abs(xt::flatten(xt::view(sol, xt::all(), 2) 
                                                + xt::view(sol, xt::all(), 3)) 
                                                - xt::flatten(xt::view(fR(max_level, i), xt::all(), 2) 
                                                            + xt::view(fR(max_level, i), xt::all(), 3)))) [0];
            
            diff_E += xt::sum(xt::abs(xt::flatten(xt::view(sol, xt::all(), 4) 
                                                + xt::view(sol, xt::all(), 5)) 
                                                - xt::flatten(xt::view(fR(max_level, i), xt::all(), 4) 
                                                            + xt::view(fR(max_level, i), xt::all(), 5)))) [0];
            
        });
    }


    return {dx * error_rho, dx * diff_rho, 
            dx * error_q, dx * diff_q,
            dx * error_E, dx * diff_E};


}

int main(int argc, char *argv[])
{
    cxxopts::Options options("lbm_d1q2_burgers",
                             "Multi resolution for a D1Q2 LBM scheme for Burgers equation");

    options.add_options()
                       ("min_level", "minimum level", cxxopts::value<std::size_t>()->default_value("2"))
                       ("max_level", "maximum level", cxxopts::value<std::size_t>()->default_value("10"))
                       ("epsilon", "maximum level", cxxopts::value<double>()->default_value("0.01"))
                       ("s", "relaxation parameter", cxxopts::value<double>()->default_value("1.0"))
                       ("log", "log level", cxxopts::value<std::string>()->default_value("warning"))
                       ("h, help", "Help");

    try
    {
        auto result = options.parse(argc, argv);

        if (result.count("help"))
            std::cout << options.help() << "\n";
        else
        {
            std::map<std::string, spdlog::level::level_enum> log_level{{"debug", spdlog::level::debug},
                                                               {"warning", spdlog::level::warn}};
            constexpr size_t dim = 1;
            using Config = mure::MRConfig<dim, 2>;

            spdlog::set_level(log_level[result["log"].as<std::string>()]);
            std::size_t min_level = 2;//result["min_level"].as<std::size_t>();
            std::size_t max_level = 9;//result["max_level"].as<std::size_t>();


            // We set some parameters according
            // to the problem.
            double sol_reg = 0.0;
            double T = 0.4;
            std::string case_name("s_d");;

            mure::Box<double, dim> box({-1}, {1});

            std::vector<double> s_vect {0.75, 1.0, 1.25, 1.5, 1.75};
            //std::vector<double> s_vect {1.5, 1.75};

            for (auto s : s_vect)   {
                std::cout<<std::endl<<"Relaxation parameter s = "<<s;

                std::string prefix (case_name + "_s_"+std::to_string(s)+"_");

                std::cout<<std::endl<<"Testing time behavior"<<std::endl;
                {
                    double eps = 1.0e-4; // This remains fixed

                    mure::Mesh<Config> mesh{box, min_level, max_level};
                    mure::Mesh<Config> meshR{box, max_level, max_level}; // This is the reference scheme

                    // Initialization
                    auto f  = init_f(mesh , 0.0);
                    auto fR = init_f(meshR, 0.0);             

                    double dx = 1.0 / (1 << max_level);
                    double dt = dx / 3.0; // Since lb = 3

                    std::size_t N = static_cast<std::size_t>(T / dt);

                    double t = 0.0;

                    std::ofstream out_time_frames;
                    
                    std::ofstream out_error_rho_exact_ref; // On the density
                    std::ofstream out_diff_rho_ref_adap;

                    std::ofstream out_error_q_exact_ref; // On the momentum
                    std::ofstream out_diff_q_ref_adap;

                    std::ofstream out_error_E_exact_ref; // On the energy
                    std::ofstream out_diff_E_ref_adap;

                    std::ofstream out_compression;

                    out_time_frames.open     ("./d1q2_3/time/"+prefix+"time.dat");

                    out_error_rho_exact_ref.open ("./d1q2_3/time/"+prefix+"error_rho.dat");
                    out_diff_rho_ref_adap.open   ("./d1q2_3/time/"+prefix+"diff_rho.dat");

                    out_error_q_exact_ref.open ("./d1q2_3/time/"+prefix+"error_q.dat");
                    out_diff_q_ref_adap.open   ("./d1q2_3/time/"+prefix+"diff_q.dat");

                    out_error_E_exact_ref.open ("./d1q2_3/time/"+prefix+"error_E.dat");
                    out_diff_E_ref_adap.open   ("./d1q2_3/time/"+prefix+"diff_E.dat");

                    out_compression.open     ("./d1q2_3/time/"+prefix+"comp.dat");


                    for (std::size_t nb_ite = 0; nb_ite < N; ++nb_ite)
                    {
                        for (std::size_t i=0; i<max_level-min_level; ++i)
                        {
                            if (coarsening(f, eps, i))
                                break;
                        }

                        for (std::size_t i=0; i<max_level-min_level; ++i)
                        {
                            if (refinement(f, eps, sol_reg, i))
                                break;
                        }

                        mure::Field<Config, int, 1> tag_leaf{"tag_leaf", mesh};
                        tag_leaf.array().fill(0);
                        mesh.for_each_cell([&](auto &cell) {
                            tag_leaf[cell] = static_cast<int>(1);
                        });
        
                        mure::Field<Config, int, 1> tag_leafR{"tag_leafR", meshR};
                        tag_leafR.array().fill(0);
                        meshR.for_each_cell([&](auto &cell) {
                            tag_leafR[cell] = static_cast<int>(1);
                        });

                        auto error = compute_error(f, fR, t);

                        out_time_frames    <<t       <<std::endl;

                        out_error_rho_exact_ref<<error[0]<<std::endl;
                        out_diff_rho_ref_adap  <<error[1]<<std::endl;

                        out_error_q_exact_ref<<error[2]<<std::endl;
                        out_diff_q_ref_adap  <<error[3]<<std::endl;

                        out_error_E_exact_ref<<error[4]<<std::endl;
                        out_diff_E_ref_adap  <<error[5]<<std::endl;

                        out_compression    <<static_cast<double>(mesh.nb_cells(mure::MeshType::cells)) 
                                           / static_cast<double>(meshR.nb_cells(mure::MeshType::cells))<<std::endl;

                        std::cout<<std::endl<<"Time = "<<t<<" Diff_h = "<<error[1]<<std::endl<<"Diff q = "<<error[3]<<std::endl<<"Diff E = "<<error[5];

                
                        one_time_step(f, tag_leaf, s);
                        one_time_step(fR, tag_leafR, s);

                        t += dt;
             
                    }

                    std::cout<<std::endl;
            
                    out_time_frames.close();

                    out_error_rho_exact_ref.close();
                    out_diff_rho_ref_adap.close();

                    out_error_q_exact_ref.close();
                    out_diff_q_ref_adap.close();

                    out_error_E_exact_ref.close();
                    out_diff_E_ref_adap.close();


                    out_compression.close();
                }
                
                std::cout<<std::endl<<"Testing eps behavior"<<std::endl;
                {
                    double eps = 1.0e-1;//0.1;
                    std::size_t N_test = 50;//50;
                    double factor = 0.60;
                    std::ofstream out_eps;
                    
                    std::ofstream out_diff_rho_ref_adap;
                    std::ofstream out_diff_q_ref_adap;
                    std::ofstream out_diff_E_ref_adap;

                    std::ofstream out_compression;

                    out_eps.open             ("./d1q2_3/eps/"+prefix+"eps.dat");

                    out_diff_rho_ref_adap.open   ("./d1q2_3/eps/"+prefix+"diff_rho.dat");
                    out_diff_q_ref_adap.open     ("./d1q2_3/eps/"+prefix+"diff_q.dat");
                    out_diff_E_ref_adap.open     ("./d1q2_3/eps/"+prefix+"diff_E.dat");

                    out_compression.open     ("./d1q2_3/eps/"+prefix+"comp.dat");

                    for (std::size_t n_test = 0; n_test < N_test; ++ n_test)    {
                        std::cout<<std::endl<<"Test "<<n_test<<" eps = "<<eps;

                        mure::Mesh<Config> mesh{box, min_level, max_level};
                        mure::Mesh<Config> meshR{box, max_level, max_level}; // This is the reference scheme

                        // Initialization
                        auto f  = init_f(mesh , 0.0);
                        auto fR = init_f(meshR, 0.0);             

                        double dx = 1.0 / (1 << max_level);
                        double dt = dx; // Since lb = 1

                        std::size_t N = static_cast<std::size_t>(T / dt);

                        double t = 0.0;

                        for (std::size_t nb_ite = 0; nb_ite < N; ++nb_ite)
                        {
                            for (std::size_t i=0; i<max_level-min_level; ++i)
                            {
                                if (coarsening(f, eps, i))
                                    break;
                            }

                            for (std::size_t i=0; i<max_level-min_level; ++i)
                            {
                                if (refinement(f, eps, sol_reg, i))
                                    break;
                            }

                            mure::Field<Config, int, 1> tag_leaf{"tag_leaf", mesh};
                            tag_leaf.array().fill(0);
                            mesh.for_each_cell([&](auto &cell) {
                                tag_leaf[cell] = static_cast<int>(1);
                            });

                            mure::Field<Config, int, 1> tag_leafR{"tag_leafR", meshR};
                            tag_leafR.array().fill(0);
                            meshR.for_each_cell([&](auto &cell) {
                                tag_leafR[cell] = static_cast<int>(1);
                            });

                            { // This is ultra important if we do not want to compute the error
                            // at each time step.
                                mure::mr_projection(f);
                                mure::mr_prediction(f); 

                                f.update_bc(); //
                                fR.update_bc();    
                            }
                        
                            

                
                            one_time_step(f, tag_leaf, s);
                            one_time_step(fR, tag_leafR, s);

            

                            t += dt;
             
                        }


                        auto error = compute_error(f, fR, 0.0);

                        std::cout<<"Diff  h= "<<error[1]<<std::endl<<"Diff q = "<<error[3]<<std::endl<<"Diff E = "<<error[5]<<std::endl;
                            
                            
                        
                        out_eps<<eps<<std::endl;

                        out_diff_rho_ref_adap<<error[1]<<std::endl;
                        out_diff_q_ref_adap<<error[3]<<std::endl;
                        out_diff_E_ref_adap<<error[5]<<std::endl;

                        out_compression<<static_cast<double>(mesh.nb_cells(mure::MeshType::cells)) 
                                           / static_cast<double>(meshR.nb_cells(mure::MeshType::cells))<<std::endl;

                        eps *= factor;
                    }
            
                    out_eps.close();  

                    out_diff_rho_ref_adap.close();
                    out_diff_q_ref_adap.close();
                    out_diff_E_ref_adap.close();

                    out_compression.close();

                }
            }
        }
    }
    
    catch (const cxxopts::OptionException &e)
    {
        std::cout << options.help() << "\n";
    }



    return 0;
}